#!/usr/bin/env python3
"""
Main Entry Point: Lightweight Gesture Control Engine for Mac & Raspberry Pi
Implements MediaPipe Hands Lite (model_complexity=0), 1€ Filter, and Kinematic FSM.
Features Real-time Motion Diagnostics and Interactive Learning Coach.
"""

import sys
import os
import subprocess

# Auto-bootstrap: If current Python interpreter lacks MediaPipe (e.g. Homebrew Python 3.14),
# automatically switch to Conda Python 3.12 where MediaPipe is installed.
try:
    import mediapipe as mp
except ImportError:
    candidates = [
        "/Users/bhargavreddy/miniconda3/bin/python3",
        "/Users/bhargavreddy/miniconda3/bin/python",
        os.path.expanduser("~/miniconda3/bin/python3"),
        os.path.expanduser("~/miniconda3/bin/python"),
        os.path.expanduser("~/anaconda3/bin/python3"),
    ]
    switched = False
    for cand in candidates:
        if os.path.exists(cand) and os.path.realpath(cand) != os.path.realpath(sys.executable):
            try:
                check = subprocess.run([cand, "-c", "import mediapipe"], capture_output=True)
                if check.returncode == 0:
                    print(f"[BOOTSTRAP] Switching from {sys.executable} to {cand}...")
                    os.execv(cand, [cand] + sys.argv)
                    switched = True
                    break
            except Exception:
                pass
    if not switched:
        print("[ERROR] MediaPipe is not installed in current Python environment.")
        sys.exit(1)

import argparse
import time
from typing import Optional, List, Dict, Any, Tuple
import cv2
import numpy as np

from gesture_engine.camera import AsyncCameraStream
from gesture_engine.fsm import VirtualMirrorGestureFSM, GestureEvent
from gesture_engine.visualizer import GestureVisualizer
from gesture_engine.synthetic import create_hand_landmarks, LandmarkMock
from gesture_engine.bridge import GestureWebSocketBridge


def select_right_hand(multi_hand_landmarks: Any, multi_handedness: Any, min_score: float = 0.7) -> Optional[Any]:
    """Return the landmark list for the first hand classified Right with sufficient
    confidence, else None.

    Only the right hand is allowed to control the mirror. MediaPipe is fed an
    already-mirrored (selfie-view) frame (see AsyncCameraStream's flip), which is
    the convention its handedness labels assume, so the labels can be trusted
    directly. Kept as a standalone, pure function so it can be unit tested without
    mocking MediaPipe/camera plumbing.
    """
    if not multi_hand_landmarks or not multi_handedness:
        return None
    for lm, handedness in zip(multi_hand_landmarks, multi_handedness):
        classification = handedness.classification[0]
        if classification.label == "Right" and classification.score >= min_score:
            return lm.landmark
    return None


class RightHandGate:
    """Adds hysteresis to right-hand selection so one noisy handedness
    classification frame (common during fast motion, hand rotation, or motion
    blur mid-gesture) doesn't drop an already-confirmed right hand.

    MediaPipe's handedness classifier is a separate, less stable model than the
    landmark/palm detector -- its confidence dips or occasionally flips label
    during exactly the fast, deliberate motion a real swipe/drag involves. With
    `max_num_hands=1` there is at most one candidate hand per frame, so once a
    right hand is locked in, a few bad classification frames in a row almost
    certainly mean the classifier wobbled, not that a different hand appeared.
    """

    def __init__(
        self,
        min_score: float = 0.7,
        lock_grace_frames: int = 12,
        relock_confirm_frames: int = 3,
    ) -> None:
        self.min_score = min_score
        self.lock_grace_frames = lock_grace_frames
        self.relock_confirm_frames = relock_confirm_frames
        self.locked = False
        self.miss_streak = 0
        self.confirm_streak = 0

    def reset(self) -> None:
        self.locked = False
        self.miss_streak = 0
        self.confirm_streak = 0

    def update(self, multi_hand_landmarks: Any, multi_handedness: Any) -> Tuple[Optional[Any], bool]:
        """Returns (landmarks_or_None, hand_rejected)."""
        if not multi_hand_landmarks:
            self.reset()
            return None, False

        lm = multi_hand_landmarks[0].landmark
        classification = multi_handedness[0].classification[0] if multi_handedness else None
        is_right = (
            classification is not None
            and classification.label == "Right"
            and classification.score >= self.min_score
        )
        is_confident_left = (
            classification is not None
            and classification.label == "Left"
            and classification.score >= self.min_score
        )

        if self.locked:
            if is_right:
                self.miss_streak = 0
                return lm, False
            self.miss_streak += 1
            if self.miss_streak <= self.lock_grace_frames:
                # Still the one hand MediaPipe is tracking -- trust continuity
                # through a brief classification wobble instead of dropping it.
                return lm, False
            self.reset()
            return None, is_confident_left

        if is_right:
            self.confirm_streak += 1
            if self.confirm_streak >= self.relock_confirm_frames:
                self.locked = True
                self.miss_streak = 0
                return lm, False
            return None, False
        self.confirm_streak = 0
        return None, is_confident_left


PRACTICE_LESSONS = [
    {
        "id": "SWIPE_RIGHT",
        "title": "LESSON 1/7: SWIPE RIGHT →",
        "tip": "Keep palm open. Flick swiftly from left to right (~0.2s).",
        "direction": "RIGHT",
        "target": "SWIPE_RIGHT",
    },
    {
        "id": "SWIPE_LEFT",
        "title": "LESSON 2/7: SWIPE LEFT ←",
        "tip": "Keep palm open. Flick swiftly from right to left.",
        "direction": "LEFT",
        "target": "SWIPE_LEFT",
    },
    {
        "id": "SWIPE_DOWN",
        "title": "LESSON 3/7: SCROLL DOWN ↓",
        "tip": "Keep palm open. Brush hand swiftly downwards.",
        "direction": "DOWN",
        "target": "SWIPE_DOWN",
    },
    {
        "id": "SWIPE_UP",
        "title": "LESSON 4/7: SCROLL UP ↑",
        "tip": "Keep palm open. Brush hand swiftly upwards.",
        "direction": "UP",
        "target": "SWIPE_UP",
    },
    {
        "id": "CLICK",
        "title": "LESSON 5/7: PINCH CLICK",
        "tip": "Bring thumb and index tips together (<0.3s) and release.",
        "direction": "CLICK",
        "target": "CLICK",
    },
    {
        "id": "HOLD",
        "title": "LESSON 6/7: STATIONARY HOLD",
        "tip": "Pinch thumb and index, and keep hand completely still for 0.5s.",
        "direction": "HOLD",
        "target": "HOLD_START",
    },
    {
        "id": "DRAG",
        "title": "LESSON 7/7: PINCH & DRAG",
        "tip": "Pinch fingers and slide hand sideways to rotate the 3D model.",
        "direction": "DRAG",
        "target": "DRAG_MOVE",
    },
]


def _pointing_hand(wrist_xy: Tuple[float, float], palm_len: float = 0.20) -> List[LandmarkMock]:
    """Synthetic hand posed with the index finger extended and the other long
    fingers curled -- the only hand shape that triggers swipe/scroll, so the
    demo/practice tour actually shows a recognized gesture."""
    lms = create_hand_landmarks(wrist_xy=wrist_xy, palm_len=palm_len, pinch_ratio=0.7)
    wx, wy = wrist_xy
    lms[12].x, lms[12].y = wx, wy - palm_len * 0.3  # middle tip, curled
    lms[16].x, lms[16].y = wx, wy - palm_len * 0.3  # ring tip, curled
    lms[20].x, lms[20].y = wx, wy - palm_len * 0.3  # pinky tip, curled
    return lms


class SyntheticGesturePlayer:
    """Simulates automated gesture cycles for demonstration and learning."""

    def __init__(self) -> None:
        self.demo_mode = False
        self.demo_start_time = 0.0

    def trigger_demo(self) -> None:
        """Start an automated 18-second tour of all 7 gestures."""
        self.demo_mode = True
        self.demo_start_time = time.perf_counter()
        print("\n[DEMO] Starting automated 7-gesture learning tour...")
        print("[DEMO] 1. SWIPE_RIGHT -> 2. SWIPE_LEFT -> 3. SWIPE_DOWN -> 4. SWIPE_UP -> 5. CLICK -> 6. HOLD -> 7. DRAG\n")

    def get_step_info(self, now: float) -> Tuple[Optional[List[LandmarkMock]], Optional[Dict[str, Any]]]:
        """Compute synthetic hand pose along programmed trajectories with lesson info."""
        if not self.demo_mode:
            return None, None

        t = now - self.demo_start_time
        cycle = t % 18.0

        # Phase 1 (0.0s - 2.5s): SWIPE_RIGHT
        if 0.4 <= cycle < 2.0:
            sub_t = cycle - 0.4
            prog = min(1.0, max(0.0, sub_t / 0.35))
            x = 0.25 + 0.48 * prog
            lms = _pointing_hand((x, 0.68))
            return lms, PRACTICE_LESSONS[0]

        # Phase 2 (2.5s - 5.0s): SWIPE_LEFT
        elif 2.8 <= cycle < 4.5:
            sub_t = cycle - 2.8
            prog = min(1.0, max(0.0, sub_t / 0.35))
            x = 0.73 - 0.48 * prog
            lms = _pointing_hand((x, 0.68))
            return lms, PRACTICE_LESSONS[1]

        # Phase 3 (5.0s - 7.5s): SWIPE_DOWN (Scroll category down)
        elif 5.3 <= cycle < 7.0:
            sub_t = cycle - 5.3
            prog = min(1.0, max(0.0, sub_t / 0.35))
            y = 0.28 + 0.25 * prog
            lms = _pointing_hand((0.50, y + 0.18))
            return lms, PRACTICE_LESSONS[2]

        # Phase 4 (7.5s - 10.0s): SWIPE_UP (Scroll category up)
        elif 7.8 <= cycle < 9.5:
            sub_t = cycle - 7.8
            prog = min(1.0, max(0.0, sub_t / 0.35))
            y = 0.53 - 0.25 * prog
            lms = _pointing_hand((0.50, y + 0.18))
            return lms, PRACTICE_LESSONS[3]

        # Phase 5 (10.0s - 12.5s): CLICK (Quick pinch & release)
        elif 10.3 <= cycle < 12.2:
            x, y = 0.50, 0.45
            sub_t = cycle - 10.3
            pinch = 0.15 if (0.3 <= sub_t <= 0.5) else 0.65
            lms = create_hand_landmarks(wrist_xy=(x, y + 0.18), pinch_ratio=pinch)
            return lms, PRACTICE_LESSONS[4]

        # Phase 6 (12.5s - 15.0s): HOLD (Stationary pinch > 0.45s)
        elif 12.8 <= cycle < 14.8:
            x, y = 0.50, 0.45
            sub_t = cycle - 12.8
            pinch = 0.15 if (0.2 <= sub_t <= 1.5) else 0.65
            lms = create_hand_landmarks(wrist_xy=(x, y + 0.18), pinch_ratio=pinch)
            return lms, PRACTICE_LESSONS[5]

        # Phase 7 (15.0s - 18.0s): DRAG (Pinch and move)
        elif 15.3 <= cycle < 17.8:
            sub_t = cycle - 15.3
            pinch = 0.15
            if sub_t < 0.3:
                x = 0.35
            else:
                prog = min(1.0, (sub_t - 0.3) / 1.5)
                x = 0.35 + 0.30 * prog
            lms = create_hand_landmarks(wrist_xy=(x, 0.68), pinch_ratio=pinch)
            return lms, PRACTICE_LESSONS[6]

        # Idle reset between phases
        lms = create_hand_landmarks(wrist_xy=(0.50, 0.68), pinch_ratio=0.70)
        curr_idx = min(len(PRACTICE_LESSONS) - 1, int(cycle / 2.6))
        return lms, PRACTICE_LESSONS[curr_idx]


def run_gesture_engine(
    mode: str = "webcam",
    camera_id: int = 0,
    width: int = 640,
    height: int = 480,
    model_complexity: int = 0,
    min_detection_confidence: float = 0.5,
    min_tracking_confidence: float = 0.4,
    headless: bool = False,
    max_frames: Optional[int] = None,
    ws_port: Optional[int] = 8765,
) -> None:
    """Run the complete perception and FSM gesture loop."""
    print("=" * 68)
    print("  ULTRA-LIGHTWEIGHT ZERO-TRAINING GESTURE CONTROL (macOS / Pi)")
    print("  KISS Architecture: MediaPipe Lite + 1€ Filter + Kinematic FSM")
    print("  Interactive Practice Coach & Ergonomic Swipe Detection Ready")
    print("=" * 68)

    bridge: Optional[GestureWebSocketBridge] = None
    if ws_port is not None and ws_port > 0:
        try:
            bridge = GestureWebSocketBridge(port=ws_port).start()
        except Exception as e:
            print(f"[WARN] WebSocket bridge initialization failed: {e}")

    stream: Optional[AsyncCameraStream] = None
    if mode != "demo":
        try:
            stream = AsyncCameraStream(
                width=width,
                height=height,
                device_index=camera_id,
                synthetic_fallback=True,
            ).start()
        except Exception as e:
            print(f"[ERROR] Could not start camera: {e}")
            print("[INFO] Switching to synthetic demo mode.")
            mode = "demo"

    # Initialize MediaPipe Hands in single-hand lite mode (1.98 MB model)
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        model_complexity=model_complexity,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )

    fsm = VirtualMirrorGestureFSM()
    visualizer = GestureVisualizer(width=width, height=height)
    demo_player = SyntheticGesturePlayer()
    right_hand_gate = RightHandGate()

    # Practice Coach State
    practice_active = False
    practice_idx = 0
    lesson_success_ts = 0.0

    if mode == "demo":
        demo_player.trigger_demo()
        visualizer.set_tutorial_lesson(PRACTICE_LESSONS[0])
    elif mode == "practice":
        practice_active = True
        visualizer.set_tutorial_lesson(PRACTICE_LESSONS[0])

    fps_count = 0
    fps_start = time.perf_counter()
    fps_val = 30.0
    frame_idx = 0

    print("\n[CONTROLS]")
    print("  't' : Toggle Interactive Practice Coach (learn swipes/gestures step-by-step)")
    print("  's' : Trigger Automated Gesture Demo Sequence")
    print("  'd' : Toggle Live Motion Diagnostics Meter (Speed, Linearity, Disp)")
    print("  'r' : Reset State Machine & Filters")
    print("  'q' : Quit\n")

    try:
        while True:
            t_now = time.perf_counter()

            # Acquire input frame
            if stream is not None and not stream.is_synthetic and mode != "demo":
                frame_bgr = stream.read()
                if frame_bgr is None:
                    time.sleep(0.005)
                    continue
                synthetic_landmarks = None
                lesson_info = None
            else:
                # Generate clean synthetic canvas
                frame_bgr = np.zeros((height, width, 3), dtype=np.uint8)
                cv2.rectangle(frame_bgr, (2, 2), (width - 2, height - 2), (25, 25, 30), -1)
                synthetic_landmarks, lesson_info = demo_player.get_step_info(t_now)
                if lesson_info is not None:
                    visualizer.set_tutorial_lesson(lesson_info)
                time.sleep(1.0 / 30.0)

            # Process Hand Perception
            hand_landmarks = None
            left_hand_rejected = False
            if synthetic_landmarks is not None:
                hand_landmarks = synthetic_landmarks
            else:
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                frame_rgb.flags.writeable = False
                results = hands.process(frame_rgb)
                frame_rgb.flags.writeable = True

                hand_landmarks, left_hand_rejected = right_hand_gate.update(
                    results.multi_hand_landmarks, results.multi_handedness
                )

            # Update State Machine
            if hand_landmarks is not None:
                event = fsm.update(hand_landmarks, t_now)
                if bridge is not None:
                    bridge.broadcast_cursor(
                        event.cursor_xy, event.state, event.pinch_ratio, event.palm_valid
                    )
                if event.action is not None:
                    visualizer.trigger_action(event.action, event.cursor_xy)
                    if bridge is not None:
                        bridge.broadcast_event(
                            event.action, event.cursor_xy, event.state
                        )
                    print(
                        f"[GESTURE EVENT] {event.action:<12} | "
                        f"State: {event.state:<13} | "
                        f"Speed: {event.current_speed:.2f} | "
                        f"Disp: {event.current_disp:.2f}x | "
                        f"Line: {int(event.current_linearity * 100)}%"
                    )

                    # Interactive Practice Coach Evaluation
                    if practice_active:
                        target = PRACTICE_LESSONS[practice_idx]["target"]
                        if event.action == target:
                            print(f"[COACH] Great job! Completed {PRACTICE_LESSONS[practice_idx]['title']}!")
                            practice_idx = (practice_idx + 1) % len(PRACTICE_LESSONS)
                            visualizer.set_tutorial_lesson(PRACTICE_LESSONS[practice_idx])
            else:
                fsm.reset()
                event = None
                if bridge is not None:
                    bridge.broadcast_cursor((0.5, 0.5), "IDLE", 1.0, False)

            # Calculate FPS
            fps_count += 1
            if (t_now - fps_start) >= 1.0:
                fps_val = fps_count / (t_now - fps_start)
                fps_count = 0
                fps_start = t_now

            # Render Visual Mirror Interface
            canvas = visualizer.render(
                frame_bgr, hand_landmarks, event, fps=fps_val, hand_rejected=left_hand_rejected
            )

            if not headless:
                cv2.imshow("Ultra-Lightweight Gesture Mirror (Mac)", canvas)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                elif key == ord("r"):
                    fsm.reset()
                    right_hand_gate.reset()
                    print("[FSM] Reset state machine.")
                elif key == ord("s"):
                    demo_player.trigger_demo()
                elif key == ord("t"):
                    practice_active = not practice_active
                    if practice_active:
                        practice_idx = 0
                        visualizer.set_tutorial_lesson(PRACTICE_LESSONS[0])
                        print("[COACH] Interactive Practice Coach Activated! Follow the on-screen instructions.")
                    else:
                        visualizer.set_tutorial_lesson(None)
                        print("[COACH] Practice Coach Closed.")
                elif key == ord("d"):
                    visualizer.show_diagnostics = not visualizer.show_diagnostics

            frame_idx += 1
            if max_frames is not None and frame_idx >= max_frames:
                break

    finally:
        print("\n[SHUTDOWN] Cleaning up resources...")
        if stream is not None:
            stream.stop()
        hands.close()
        cv2.destroyAllWindows()
        print("[SHUTDOWN] Gesture engine stopped cleanly.")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ultra-Lightweight Zero-Training Gesture Control for Mac")
    parser.add_argument(
        "--mode",
        choices=["webcam", "demo", "practice"],
        default="webcam",
        help="Run live with webcam, automated demo, or practice coach (default: webcam)",
    )
    parser.add_argument("--camera-id", type=int, default=0, help="Camera device index (default: 0)")
    parser.add_argument("--width", type=int, default=640, help="Frame width (default: 640)")
    parser.add_argument("--height", type=int, default=480, help="Frame height (default: 480)")
    parser.add_argument("--headless", action="store_true", help="Run without graphical display")
    parser.add_argument("--max-frames", type=int, default=None, help="Limit maximum frames to process")
    parser.add_argument(
        "--ws-port",
        type=int,
        default=8765,
        help="WebSocket bridge port for website/webpage integration (default: 8765, set 0 to disable)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    run_gesture_engine(
        mode=args.mode,
        camera_id=args.camera_id,
        width=args.width,
        height=args.height,
        headless=args.headless,
        max_frames=args.max_frames,
        ws_port=args.ws_port if args.ws_port > 0 else None,
    )
