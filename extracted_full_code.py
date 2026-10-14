import math
import time
import threading
from collections import deque
from dataclasses import dataclass
from typing import Optional, Tuple, Deque

import cv2
import numpy as np
import mediapipe as mp

# Attempt to load Picamera2 for CSI ribbon cameras on Raspberry Pi OS Bookworm;
# fall back gracefully to OpenCV VideoCapture for USB webcams.
try:
    from picamera2 import Picamera2
    PICAMERA2_AVAILABLE = True
except ImportError:
    PICAMERA2_AVAILABLE = False


# =====================================================================
# 1. SIGNAL STABILIZATION: 1 EURO FILTER (SPEED-ADAPTIVE LOW-PASS)
# =====================================================================
class OneEuroFilter1D:
    """First-order speed-adaptive low-pass filter to eliminate jitter without lag."""

    def __init__(
        self,
        min_cutoff: float = 1.2,
        beta: float = 0.35,
        d_cutoff: float = 1.0,
    ) -> None:
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self.x_prev: Optional[float] = None
        self.dx_prev: float = 0.0
        self.t_prev: Optional[float] = None

    @staticmethod
    def _alpha(cutoff: float, dt: float) -> float:
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def reset(self) -> None:
        self.x_prev = None
        self.dx_prev = 0.0
        self.t_prev = None

    def filter(self, x: float, t: float) -> float:
        if self.t_prev is None or self.x_prev is None:
            self.x_prev = x
            self.t_prev = t
            return x

        dt = t - self.t_prev
        if dt <= 1e-6:
            return self.x_prev

        # Estimate and smooth the signal derivative (velocity)
        dx = (x - self.x_prev) / dt
        a_d = self._alpha(self.d_cutoff, dt)
        dx_hat = a_d * dx + (1.0 - a_d) * self.dx_prev

        # Dynamically adjust cutoff frequency based on speed
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = self._alpha(cutoff, dt)
        x_hat = a * x + (1.0 - a) * self.x_prev

        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = t
        return x_hat


class Point2DEuroFilter:
    """Paired 2D One Euro Filter for normalized (x, y) coordinates."""

    def __init__(self, min_cutoff: float = 1.2, beta: float = 0.35) -> None:
        self.fx = OneEuroFilter1D(min_cutoff=min_cutoff, beta=beta)
        self.fy = OneEuroFilter1D(min_cutoff=min_cutoff, beta=beta)

    def reset(self) -> None:
        self.fx.reset()
        self.fy.reset()

    def filter(self, x: float, y: float, t: float) -> Tuple[float, float]:
        return self.fx.filter(x, t), self.fy.filter(y, t)


# =====================================================================
# 2. ASYNCHRONOUS CAMERA STREAM (DECOUPLED I/O THREAD)
# =====================================================================
class AsyncCameraStream:
    """Daemon thread camera reader supporting Picamera2 and USB Webcams."""

    def __init__(self, width: int = 640, height: int = 480, use_picam: bool = True) -> None:
        self.width = width
        self.height = height
        self.use_picam = use_picam and PICAMERA2_AVAILABLE
        self.frame: Optional[np.ndarray] = None
        self.running = False
        self.lock = threading.Lock()

        if self.use_picam:
            self.picam2 = Picamera2()
            config = self.picam2.create_video_configuration(
                main={"size": (self.width, self.height), "format": "RGB888"}
            )
            self.picam2.configure(config)
            self.picam2.start()
        else:
            self.cap = cv2.VideoCapture(0)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.cap.set(cv2.CAP_PROP_FPS, 30)

    def start(self) -> "AsyncCameraStream":
        self.running = True
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()
        return self

    def _update(self) -> None:
        while self.running:
            if self.use_picam:
                raw = self.picam2.capture_array()
                # Picamera2 RGB888 outputs BGR-ordered bytes in OpenCV conventions on Pi
                bgr = raw
            else:
                ret, bgr = self.cap.read()
                if not ret:
                    time.sleep(0.005)
                    continue

            # Horizontal flip for natural mirror-mode interaction
            flipped = cv2.flip(bgr, 1)
            with self.lock:
                self.frame = flipped

    def read(self) -> Optional[np.ndarray]:
        with self.lock:
            return self.frame.copy() if self.frame is not None else None

    def stop(self) -> None:
        self.running = False
        if hasattr(self, "thread"):
            self.thread.join(timeout=1.0)
        if self.use_picam:
            self.picam2.stop()
        else:
            self.cap.release()


# =====================================================================
# 3. ZERO-TRAINING KINEMATIC FINITE STATE MACHINE (FSM)
# =====================================================================
@dataclass
class GestureEvent:
    state: str                       # 'IDLE', 'HOVER', 'PINCH_PENDING', 'HOLD', 'DRAG'
    action: Optional[str]            # 'CLICK', 'HOLD_START', 'DRAG_MOVE', 'DRAG_END', 'SWIPE_*'
    cursor_xy: Tuple[float, float]   # Normalized (x, y) in [0.0, 1.0]
    pinch_ratio: float               # Scale-invariant pinch distance


class VirtualMirrorGestureFSM:
    """Deterministic kinematic state machine for spatial & temporal gestures."""

    def __init__(self) -> None:
        # Pinch hysteresis thresholds (normalized by Wrist-to-Middle-MCP palm length)
        self.pinch_engage_thresh = 0.32
        self.pinch_release_thresh = 0.45

        # Spatial and temporal thresholds for Click, Hold, and Drag
        self.deadzone_palm_ratio = 0.25
        self.click_min_sec = 0.04
        self.click_max_sec = 0.30
        self.hold_trigger_sec = 0.45

        # Temporal swipe parameters
        self.swipe_window = 7
        self.swipe_min_palm_disp = 1.60
        self.swipe_min_screen_disp = 0.18
        self.swipe_min_velocity = 0.65
        self.swipe_linearity_min = 0.80
        self.swipe_axis_ratio = 1.80
        self.swipe_cooldown_sec = 0.45

        # Internal state variables
        self.state = "IDLE"
        self.pinch_start_time = 0.0
        self.anchor_xy = (0.5, 0.5)
        self.last_swipe_time = 0.0
        self.palm_history: Deque[Tuple[float, float, float]] = deque(maxlen=self.swipe_window)

        # 1 Euro Filters for cursor and pinch tips
        self.thumb_filter = Point2DEuroFilter(min_cutoff=1.2, beta=0.35)
        self.index_filter = Point2DEuroFilter(min_cutoff=1.2, beta=0.35)
        self.palm_filter = Point2DEuroFilter(min_cutoff=1.5, beta=0.40)

    def reset(self) -> None:
        self.state = "IDLE"
        self.palm_history.clear()
        self.thumb_filter.reset()
        self.index_filter.reset()
        self.palm_filter.reset()

    def update(self, landmarks, now: float) -> GestureEvent:
        # Extract key anatomical landmarks: Wrist (0), Thumb Tip (4), Index Tip (8), Middle MCP (9)
        w0 = landmarks[0]
        t4 = landmarks[4]
        i8 = landmarks[8]
        m9 = landmarks[9]

        # Stabilize coordinates with 1 Euro Filter
        tx, ty = self.thumb_filter.filter(t4.x, t4.y, now)
        ix, iy = self.index_filter.filter(i8.x, i8.y, now)
        px, py = self.palm_filter.filter(m9.x, m9.y, now)

        # Compute scale-invariant palm reference length (Wrist 0 to Middle MCP 9)
        palm_scale = math.hypot(m9.x - w0.x, m9.y - w0.y)
        palm_scale = max(palm_scale, 1e-4)

        # Cursor is anchored to the stabilized midpoint between Thumb Tip and Index Tip
        cursor_xy = ((tx + ix) * 0.5, (ty + iy) * 0.5)
        pinch_dist = math.hypot(ix - tx, iy - ty)
        pinch_ratio = pinch_dist / palm_scale

        action: Optional[str] = None

        # --- STATE MACHINE TRANSITIONS ---
        if self.state in ("IDLE", "HOVER"):
            self.state = "HOVER"
            if pinch_ratio < self.pinch_engage_thresh:
                # Transition to PINCH_PENDING
                self.state = "PINCH_PENDING"
                self.pinch_start_time = now
                self.anchor_xy = cursor_xy
                self.palm_history.clear()
            else:
                # Track palm trajectory for open-hand temporal swipes
                self.palm_history.append((px, py, now))
                action = self._evaluate_swipe(palm_scale, now)

        elif self.state == "PINCH_PENDING":
            elapsed = now - self.pinch_start_time
            disp_ratio = math.hypot(
                cursor_xy[0] - self.anchor_xy[0],
                cursor_xy[1] - self.anchor_xy[1]
            ) / palm_scale

            if pinch_ratio > self.pinch_release_thresh:
                # Pinch released: check if valid CLICK
                if self.click_min_sec <= elapsed < self.click_max_sec and disp_ratio < self.deadzone_palm_ratio:
                    action = "CLICK"
                self.state = "HOVER"
            elif disp_ratio >= self.deadzone_palm_ratio:
                # Moved outside deadzone while pinched -> DRAG
                self.state = "DRAG"
                action = "DRAG_START"
            elif elapsed >= self.hold_trigger_sec:
                # Held stationary inside deadzone -> HOLD
                self.state = "HOLD"
                action = "HOLD_START"

        elif self.state == "HOLD":
            disp_ratio = math.hypot(
                cursor_xy[0] - self.anchor_xy[0],
                cursor_xy[1] - self.anchor_xy[1]
            ) / palm_scale

            if pinch_ratio > self.pinch_release_thresh:
                self.state = "HOVER"
                action = "HOLD_END"
            elif disp_ratio >= self.deadzone_palm_ratio:
                # Allows seamless transition from Hold into Drag
                self.state = "DRAG"
                action = "DRAG_START"

        elif self.state == "DRAG":
            if pinch_ratio > self.pinch_release_thresh:
                self.state = "HOVER"
                action = "DRAG_END"
            else:
                action = "DRAG_MOVE"

        return GestureEvent(
            state=self.state,
            action=action,
            cursor_xy=cursor_xy,
            pinch_ratio=pinch_ratio,
        )

    def _evaluate_swipe(self, palm_scale: float, now: float) -> Optional[str]:
        if len(self.palm_history) < self.swipe_window:
            return None
        if (now - self.last_swipe_time) < self.swipe_cooldown_sec:
            return None

        x0, y0, t0 = self.palm_history[0]
        x1, y1, t1 = self.palm_history[-1]
        dt = t1 - t0
        if dt <= 1e-4:
            return None

        dx = x1 - x0
        dy = y1 - y0
        net_disp = math.hypot(dx, dy)

        # Gate 1: Minimum screen and scale-invariant displacement & velocity
        if net_disp < self.swipe_min_screen_disp:
            return None
        if (net_disp / palm_scale) < self.swipe_min_palm_disp:
            return None
        if (net_disp / dt) < self.swipe_min_velocity:
            return None

        # Gate 2: Path linearity (straightness) check
        path_length = 0.0
        for k in range(1, len(self.palm_history)):
            xa, ya, _ = self.palm_history[k - 1]
            xb, yb, _ = self.palm_history[k]
            path_length += math.hypot(xb - xa, yb - ya)

        if path_length <= 1e-5 or (net_disp / path_length) < self.swipe_linearity_min:
            return None

        # Gate 3: Dominant axis disambiguation
        swipe_cmd: Optional[str] = None
        if abs(dx) > self.swipe_axis_ratio * abs(dy):
            swipe_cmd = "SWIPE_RIGHT" if dx > 0 else "SWIPE_LEFT"
        elif abs(dy) > self.swipe_axis_ratio * abs(dx):
            swipe_cmd = "SWIPE_DOWN" if dy > 0 else "SWIPE_UP"

        if swipe_cmd is not None:
            self.last_swipe_time = now
            self.palm_history.clear()

        return swipe_cmd


# =====================================================================
# 4. MAIN PERCEPTION LOOP FOR RASPBERRY PI VIRTUAL MIRROR
# =====================================================================
def run_virtual_mirror_gesture_engine() -> None:
    W, H = 640, 480
    stream = AsyncCameraStream(width=W, height=H, use_picam=True).start()

    # Configure MediaPipe Hands in ultra-lightweight single-hand mode (1.98 MB model)
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        model_complexity=0,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.5,
    )

    fsm = VirtualMirrorGestureFSM()
    last_action_label = "NONE"
    last_action_ts = 0.0

    try:
        while True:
            frame_bgr = stream.read()
            if frame_bgr is None:
                time.sleep(0.005)
                continue

            t_now = time.perf_counter()
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            frame_rgb.flags.writeable = False
            results = hands.process(frame_rgb)
            frame_rgb.flags.writeable = True

            if results.multi_hand_landmarks:
                hand_lms = results.multi_hand_landmarks[0].landmark
                event = fsm.update(hand_lms, t_now)

                if event.action is not None:
                    last_action_label = event.action
                    last_action_ts = t_now
                    # Hook for Virtual Mirror UI event bus (e.g., WebSocket / OSC / PyQt)
                    print(f"[GESTURE EVENT] {event.action} | State: {event.state} | Pos: {event.cursor_xy}")

                # Render lightweight cursor for visual feedback
                cx, cy = int(event.cursor_xy[0] * W), int(event.cursor_xy[1] * H)
                color = (0, 255, 0) if event.state in ("PINCH_PENDING", "DRAG", "HOLD") else (255, 180, 0)
                cv2.circle(frame_bgr, (cx, cy), 10, color, -1)
            else:
                fsm.reset()

            # Display active state and most recent triggered gesture
            if (t_now - last_action_ts) > 1.2:
                last_action_label = "NONE"

            cv2.putText(
                frame_bgr,
                f"State: {fsm.state} | Event: {last_action_label}",
                (15, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (255, 255, 255),
                2,
            )
            cv2.imshow("Virtual Mirror Gesture Engine", frame_bgr)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        stream.stop()
        hands.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    run_virtual_mirror_gesture_engine()
