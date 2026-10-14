#!/usr/bin/env python3
"""
Comprehensive Test Suite for Ultra-Lightweight Gesture Control
Verifies 1€ Filter, Scale Invariance, and FSM Transitions for all 7 Gestures.
"""

import sys
import os
import subprocess

try:
    import mediapipe
except ImportError:
    candidates = [
        "/Users/bhargavreddy/miniconda3/bin/python3",
        "/Users/bhargavreddy/miniconda3/bin/python",
        os.path.expanduser("~/miniconda3/bin/python3"),
        os.path.expanduser("~/miniconda3/bin/python"),
    ]
    for cand in candidates:
        if os.path.exists(cand) and os.path.realpath(cand) != os.path.realpath(sys.executable):
            try:
                check = subprocess.run([cand, "-c", "import mediapipe"], capture_output=True)
                if check.returncode == 0:
                    os.execv(cand, [cand] + sys.argv)
            except Exception:
                pass

import unittest
from gesture_engine.filter import OneEuroFilter1D, Point2DEuroFilter
from gesture_engine.fsm import VirtualMirrorGestureFSM
from gesture_engine.synthetic import create_hand_landmarks
from main import select_right_hand, RightHandGate


def pointing_hand(wrist_xy, palm_len: float = 0.15, pinch_ratio: float = 0.7):
    """A synthetic hand with the index finger extended and the other long
    fingers curled -- the only hand shape the FSM recognizes for swipe/scroll.
    """
    lms = create_hand_landmarks(wrist_xy=wrist_xy, palm_len=palm_len, pinch_ratio=pinch_ratio)
    wx, wy = wrist_xy
    # Index is already extended in create_hand_landmarks's default pose (tip far
    # from wrist). Curl middle/ring/pinky tips back toward the wrist.
    lms[12].x, lms[12].y = wx, wy - palm_len * 0.3  # middle tip, curled
    lms[16].x, lms[16].y = wx, wy - palm_len * 0.3  # ring tip, curled
    lms[20].x, lms[20].y = wx, wy - palm_len * 0.3  # pinky tip, curled
    return lms


class TestOneEuroFilter(unittest.TestCase):
    """Unit tests for the 1€ Filter."""

    def test_filter_initialization(self) -> None:
        f = OneEuroFilter1D(min_cutoff=1.2, beta=0.35)
        val = f.filter(10.0, 1.0)
        self.assertEqual(val, 10.0)

    def test_filter_adaptation_stationary_vs_dynamic(self) -> None:
        """
        Verify that stationary input is heavily smoothed (low cutoff),
        while high velocity motion adapts cutoff upward to eliminate lag.
        """
        f = OneEuroFilter1D(min_cutoff=1.0, beta=1.0)
        t = 0.0
        dt = 1.0 / 30.0

        # Feed steady baseline
        for _ in range(10):
            t += dt
            f.filter(0.0, t)

        # Small step (stationary noise)
        small_step_val = f.filter(0.05, t + dt)
        self.assertLess(small_step_val, 0.04)

        # High-speed step (rapid motion)
        f_fast = OneEuroFilter1D(min_cutoff=1.0, beta=1.0)
        t_fast = 0.0
        for _ in range(10):
            t_fast += dt
            f_fast.filter(0.0, t_fast)

        # Large sudden leap
        large_step_val = f_fast.filter(100.0, t_fast + dt)
        self.assertGreater(large_step_val, 50.0)


class TestKinematicFSM(unittest.TestCase):
    """Comprehensive test cases for the kinematic state machine."""

    def setUp(self) -> None:
        self.fsm = VirtualMirrorGestureFSM()

    def test_scale_invariance_far_vs_close(self) -> None:
        """
        Ensure pinch ratio is scale-invariant whether hand is far (small palm) or close (large palm).
        """
        # User standing far from mirror (palm length 0.12)
        lms_far = create_hand_landmarks(wrist_xy=(0.5, 0.6), palm_len=0.12, pinch_ratio=0.25)
        event_far = self.fsm.update(lms_far, 1.0)
        self.assertAlmostEqual(event_far.pinch_ratio, 0.25, delta=0.03)

        self.fsm.reset()

        # User standing close to mirror (palm length 0.35)
        lms_close = create_hand_landmarks(wrist_xy=(0.5, 0.6), palm_len=0.35, pinch_ratio=0.25)
        event_close = self.fsm.update(lms_close, 1.0)
        self.assertAlmostEqual(event_close.pinch_ratio, 0.25, delta=0.03)

    def test_click_gesture(self) -> None:
        """
        Test CLICK: Pinch engaged (<0.32), held in deadzone for ~0.12s, then released (>0.45).
        """
        t = 1.0
        dt = 1.0 / 30.0

        # Step 1: Pre-settle open hand in HOVER
        for _ in range(5):
            t += dt
            ev = self.fsm.update(create_hand_landmarks(wrist_xy=(0.5, 0.6), pinch_ratio=0.7), t)
        self.assertEqual(ev.state, "HOVER")

        # Step 2: Pinch engaged (touching fingers reach < 0.24 over continuous frames)
        for _ in range(8):
            t += dt
            ev = self.fsm.update(create_hand_landmarks(wrist_xy=(0.5, 0.6), pinch_ratio=0.10), t)
        self.assertEqual(ev.state, "PINCH_PENDING")

        # Step 3: Hold pinch for 4 frames (elapsed within [0.08, 0.35]s window)
        for _ in range(4):
            t += dt
            ev = self.fsm.update(create_hand_landmarks(wrist_xy=(0.5, 0.6), pinch_ratio=0.10), t)
            self.assertEqual(ev.state, "PINCH_PENDING")

        # Step 4: Release pinch over continuous frames
        actions = []
        for _ in range(6):
            t += dt
            ev = self.fsm.update(create_hand_landmarks(wrist_xy=(0.5, 0.6), pinch_ratio=0.70), t)
            if ev.action:
                actions.append(ev.action)

        self.assertIn("CLICK", actions)
        self.assertEqual(ev.state, "HOVER")

    def test_hold_gesture(self) -> None:
        """
        Test HOLD: Pinch engaged and held stationary inside deadzone for >0.45s.
        """
        t = 10.0
        dt = 1.0 / 30.0

        # Pre-settle
        for _ in range(5):
            t += dt
            self.fsm.update(create_hand_landmarks(wrist_xy=(0.5, 0.6), pinch_ratio=0.7), t)

        # Pinch engaged
        for _ in range(8):
            t += dt
            ev = self.fsm.update(create_hand_landmarks(wrist_xy=(0.5, 0.6), pinch_ratio=0.10), t)
        self.assertEqual(ev.state, "PINCH_PENDING")

        # Stay stationary for 16 frames (16 * 0.0333 = 0.533s > 0.45s trigger)
        actions = []
        for _ in range(16):
            t += dt
            ev = self.fsm.update(create_hand_landmarks(wrist_xy=(0.5, 0.6), pinch_ratio=0.10), t)
            if ev.action is not None:
                actions.append(ev.action)

        self.assertIn("HOLD_START", actions)
        self.assertEqual(ev.state, "HOLD")

        # Release pinch
        release_actions = []
        for _ in range(6):
            t += dt
            ev = self.fsm.update(create_hand_landmarks(wrist_xy=(0.5, 0.6), pinch_ratio=0.70), t)
            if ev.action:
                release_actions.append(ev.action)

        self.assertIn("HOLD_END", release_actions)
        self.assertEqual(ev.state, "HOVER")

    def test_drag_gesture(self) -> None:
        """
        Test DRAG: Pinch engaged and moved beyond deadzone ratio (>= 0.25 palm scale).
        """
        t = 20.0
        dt = 1.0 / 30.0
        palm_len = 0.20

        # Pre-settle at x=0.40
        for _ in range(5):
            t += dt
            self.fsm.update(create_hand_landmarks(wrist_xy=(0.40, 0.6), palm_len=palm_len, pinch_ratio=0.7), t)

        # Pinch engaged at x=0.40
        for _ in range(8):
            t += dt
            ev = self.fsm.update(create_hand_landmarks(wrist_xy=(0.40, 0.6), palm_len=palm_len, pinch_ratio=0.10), t)
        self.assertEqual(ev.state, "PINCH_PENDING")

        # Move hand horizontally by 0.15 (0.15 / 0.20 = 0.75 palm ratio > 0.25 deadzone)
        drag_actions = []
        for i in range(10):
            t += dt
            x = 0.40 + (0.15 * (i + 1) / 10.0)
            ev = self.fsm.update(create_hand_landmarks(wrist_xy=(x, 0.6), palm_len=palm_len, pinch_ratio=0.20), t)
            if ev.action:
                drag_actions.append(ev.action)

        self.assertIn("DRAG_START", drag_actions)
        self.assertIn("DRAG_MOVE", drag_actions)
        self.assertEqual(ev.state, "DRAG")

        # Release pinch
        rel_actions = []
        for _ in range(6):
            t += dt
            ev = self.fsm.update(create_hand_landmarks(wrist_xy=(0.55, 0.6), palm_len=palm_len, pinch_ratio=0.70), t)
            if ev.action:
                rel_actions.append(ev.action)

        self.assertIn("DRAG_END", rel_actions)
        self.assertEqual(ev.state, "HOVER")

    def test_directional_swipes(self) -> None:
        """
        Test directional swipes: SWIPE_RIGHT, SWIPE_LEFT, SWIPE_DOWN, SWIPE_UP.
        Verifies 4-gate kinematic checks and cooldown-based rebound suppression.
        """
        dt = 1.0 / 30.0
        palm_len = 0.15

        # --- Test SWIPE_RIGHT ---
        t = 30.0
        self.fsm.reset()
        swipe_events = []
        for i in range(7):
            t += dt
            x = 0.20 + (0.35 * (i / 6.0))
            lms = pointing_hand((x, 0.60), palm_len=palm_len)
            ev = self.fsm.update(lms, t)
            if ev.action:
                swipe_events.append(ev.action)

        self.assertIn("SWIPE_RIGHT", swipe_events)

        # Smooth return stroke to x=0.70 during 0.5s cooldown (rebound check)
        rebound_events = []
        for _ in range(16):
            t += dt
            ev = self.fsm.update(pointing_hand((0.70, 0.60), palm_len=palm_len), t)
            if ev.action:
                rebound_events.append(ev.action)
        self.assertEqual(len(rebound_events), 0, "Cooldown failed to suppress return stroke!")

        # --- Test SWIPE_LEFT ---
        swipe_left_events = []
        for i in range(7):
            t += dt
            x = 0.70 - (0.35 * (i / 6.0))
            lms = pointing_hand((x, 0.60), palm_len=palm_len)
            ev = self.fsm.update(lms, t)
            if ev.action:
                swipe_left_events.append(ev.action)

        self.assertIn("SWIPE_LEFT", swipe_left_events)

        # Return stroke to y=0.35 during cooldown
        for _ in range(16):
            t += dt
            self.fsm.update(pointing_hand((0.50, 0.35), palm_len=palm_len), t)

        # --- Test SWIPE_DOWN ---
        swipe_down_events = []
        for i in range(7):
            t += dt
            y = 0.35 + (0.35 * (i / 6.0))
            lms = pointing_hand((0.50, y), palm_len=palm_len)
            ev = self.fsm.update(lms, t)
            if ev.action:
                swipe_down_events.append(ev.action)

        self.assertIn("SWIPE_DOWN", swipe_down_events)

        # Return stroke to y=0.75 during cooldown
        for _ in range(16):
            t += dt
            self.fsm.update(pointing_hand((0.50, 0.75), palm_len=palm_len), t)

        # --- Test SWIPE_UP ---
        swipe_up_events = []
        for i in range(7):
            t += dt
            y = 0.75 - (0.35 * (i / 6.0))
            lms = pointing_hand((0.50, y), palm_len=palm_len)
            ev = self.fsm.update(lms, t)
            if ev.action:
                swipe_up_events.append(ev.action)

        self.assertIn("SWIPE_UP", swipe_up_events)

    def test_false_pinch_rejection_photo_case(self) -> None:
        """
        Test that 2D overlap where thumb is tucked behind/under fingers (3D z difference)
        does NOT trigger false pinch or drag.
        """
        self.fsm.reset()
        t = 50.0
        dt = 1.0 / 30.0

        # Create hand where thumb (4) and index (8) have small 2D distance but large 3D depth difference
        lms = create_hand_landmarks(wrist_xy=(0.50, 0.60), pinch_ratio=0.7)
        # Manually offset thumb in z (depth) behind index
        lms[4].x = lms[8].x + 0.01
        lms[4].y = lms[8].y + 0.01
        lms[4].z = -0.10   # 10cm behind in depth
        lms[8].z = 0.04    # in front

        for _ in range(10):
            t += dt
            ev = self.fsm.update(lms, t)
            # Must remain in HOVER, never enter PINCH_PENDING or DRAG
            self.assertEqual(ev.state, "HOVER", f"Falsely entered {ev.state} during 2D optical overlap!")

    def test_pinch_requires_consecutive_confirmation_frames(self) -> None:
        """
        A clean, pinch-eligible pose must be seen for several consecutive frames
        (pinch_confirm_frames) before PINCH_PENDING is trusted -- not on the very
        first qualifying frame.
        """
        self.fsm.reset()
        t = 120.0
        dt = 1.0 / 30.0

        for i in range(self.fsm.pinch_confirm_frames - 1):
            t += dt
            lms = create_hand_landmarks(wrist_xy=(0.50, 0.60), pinch_ratio=0.10)
            ev = self.fsm.update(lms, t)
            self.assertEqual(ev.state, "HOVER", f"Engaged PINCH_PENDING too early on frame {i}!")

        t += dt
        lms = create_hand_landmarks(wrist_xy=(0.50, 0.60), pinch_ratio=0.10)
        ev = self.fsm.update(lms, t)
        self.assertEqual(ev.state, "PINCH_PENDING", "Failed to confirm pinch after enough clear frames!")

    def test_single_noisy_frame_resets_pinch_confirmation(self) -> None:
        """
        A single noisy/ambiguous-depth frame sandwiched between clean pinch frames
        must reset the confirmation counter, not just pause it.
        """
        self.fsm.reset()
        t = 125.0
        dt = 1.0 / 30.0

        # One clean candidate frame.
        t += dt
        ev = self.fsm.update(create_hand_landmarks(wrist_xy=(0.50, 0.60), pinch_ratio=0.10), t)
        self.assertEqual(ev.state, "HOVER")

        # One noisy frame: thumb/index close in 2D but far apart in depth (fails coplanar).
        t += dt
        noisy = create_hand_landmarks(wrist_xy=(0.50, 0.60), pinch_ratio=0.10)
        noisy[4].z = -0.10
        noisy[8].z = 0.04
        ev = self.fsm.update(noisy, t)
        self.assertEqual(ev.state, "HOVER")

        # Now a fresh run of clean frames -- must take the full confirmation count again,
        # not immediately engage as if the earlier clean frame still counted.
        for i in range(self.fsm.pinch_confirm_frames - 1):
            t += dt
            ev = self.fsm.update(create_hand_landmarks(wrist_xy=(0.50, 0.60), pinch_ratio=0.10), t)
            self.assertEqual(ev.state, "HOVER", f"Confirmation counter wasn't reset by the noisy frame (i={i})!")

        t += dt
        ev = self.fsm.update(create_hand_landmarks(wrist_xy=(0.50, 0.60), pinch_ratio=0.10), t)
        self.assertEqual(ev.state, "PINCH_PENDING")

    def test_palm_size_ok_relaxed_threshold(self) -> None:
        """
        Click/pinch engagement should now succeed at a smaller apparent palm size
        (further from camera) than the original 0.06 floor.
        """
        self.fsm.reset()
        t = 128.0
        dt = 1.0 / 30.0
        palm_len = 0.05  # between the new 0.045 floor and the old 0.06 floor

        ev = None
        for _ in range(self.fsm.pinch_confirm_frames):
            t += dt
            lms = create_hand_landmarks(wrist_xy=(0.50, 0.60), palm_len=palm_len, pinch_ratio=0.10)
            ev = self.fsm.update(lms, t)

        self.assertTrue(ev.palm_valid, "Distant hand incorrectly flagged as invalid!")
        self.assertEqual(ev.state, "PINCH_PENDING", "Distant hand failed to engage pinch at relaxed threshold!")

    def test_distant_hand_still_requires_confirmation(self) -> None:
        """
        At the new, more permissive distance threshold, a noisy frame must still
        reset the confirmation counter -- distance alone shouldn't bypass it.
        """
        self.fsm.reset()
        t = 129.0
        dt = 1.0 / 30.0
        palm_len = 0.05

        t += dt
        ev = self.fsm.update(create_hand_landmarks(wrist_xy=(0.50, 0.60), palm_len=palm_len, pinch_ratio=0.10), t)
        self.assertEqual(ev.state, "HOVER")

        t += dt
        noisy = create_hand_landmarks(wrist_xy=(0.50, 0.60), palm_len=palm_len, pinch_ratio=0.10)
        noisy[4].z = -0.03
        noisy[8].z = 0.015
        ev = self.fsm.update(noisy, t)
        self.assertEqual(ev.state, "HOVER", "Noisy frame at distance was not rejected!")

        for _ in range(self.fsm.pinch_confirm_frames - 1):
            t += dt
            ev = self.fsm.update(
                create_hand_landmarks(wrist_xy=(0.50, 0.60), palm_len=palm_len, pinch_ratio=0.10), t
            )
            self.assertEqual(ev.state, "HOVER")

        t += dt
        ev = self.fsm.update(create_hand_landmarks(wrist_xy=(0.50, 0.60), palm_len=palm_len, pinch_ratio=0.10), t)
        self.assertEqual(ev.state, "PINCH_PENDING")

    def test_swipe_breakout_from_drag(self) -> None:
        """
        Test that if user is in DRAG state and performs a rapid swipe stroke,
        the FSM immediately breaks out of DRAG and fires the SWIPE.
        """
        self.fsm.reset()
        t = 60.0
        dt = 1.0 / 30.0
        palm_len = 0.15

        # Engage pinch
        for _ in range(8):
            t += dt
            self.fsm.update(create_hand_landmarks(wrist_xy=(0.30, 0.60), palm_len=palm_len, pinch_ratio=0.10), t)

        # Gentle drag motion
        for i in range(5):
            t += dt
            ev = self.fsm.update(create_hand_landmarks(wrist_xy=(0.30 + 0.015 * (i+1), 0.60), palm_len=palm_len, pinch_ratio=0.10), t)
        self.assertEqual(ev.state, "DRAG")

        # Now execute a rapid swipe stroke right (pointing pose, the only gesture
        # that can trigger a swipe)
        swipe_action = None
        for i in range(7):
            t += dt
            x = 0.38 + (0.30 * (i / 6.0))
            ev = self.fsm.update(pointing_hand((x, 0.60), palm_len=palm_len, pinch_ratio=0.50), t)
            if ev.action in ("SWIPE_RIGHT", "SWIPE_LEFT", "SWIPE_DOWN", "SWIPE_UP"):
                swipe_action = ev.action

        self.assertEqual(swipe_action, "SWIPE_RIGHT", "Failed to break out of DRAG on rapid swipe stroke!")

    def test_fast_short_flick_registers_swipe(self) -> None:
        """
        A short, fast flick (high velocity, but less travel than a full deliberate
        swipe) must still register. The fixed 0.90x-palm-length displacement floor
        used to reject this; the velocity-aware floor should now accept it.
        """
        self.fsm.reset()
        t = 80.0
        dt = 1.0 / 30.0
        palm_len = 0.15

        events = []
        for i in range(4):
            t += dt
            x = 0.30 + 0.12 * (i / 3.0)  # net disp 0.12 -> 0.8x palm_len (< 0.90, > 0.45)
            lms = pointing_hand((x, 0.60), palm_len=palm_len)
            ev = self.fsm.update(lms, t)
            if ev.action:
                events.append(ev.action)

        self.assertIn("SWIPE_RIGHT", events, "Fast short flick failed to register as a swipe!")

    def test_moderate_speed_small_displacement_still_rejected(self) -> None:
        """
        Regression guard: a moderate-speed, genuinely small motion (not a deliberate
        flick) must still be rejected, even with the relaxed fast-flick floors.
        """
        self.fsm.reset()
        t = 90.0
        dt = 1.0 / 30.0
        palm_len = 0.25

        events = []
        for i in range(5):
            t += dt
            x = 0.30 + 0.10 * (i / 4.0)  # net disp 0.10 -> 0.4x palm_len, velocity ~0.75
            lms = pointing_hand((x, 0.60), palm_len=palm_len)
            ev = self.fsm.update(lms, t)
            if ev.action:
                events.append(ev.action)

        self.assertEqual(events, [], f"Small, casual motion falsely registered as a swipe: {events}")

    def test_swipe_no_false_pinch_during_and_after_stroke(self) -> None:
        """
        During and immediately after a qualifying swipe stroke, a transient dip in
        pinch_ratio (motion blur / landmark jitter) must not be mistaken for a real
        pinch engagement.
        """
        self.fsm.reset()
        t = 100.0
        dt = 1.0 / 30.0
        palm_len = 0.15

        states_seen = set()
        actions_seen = []
        for i in range(7):
            t += dt
            x = 0.20 + (0.35 * (i / 6.0))
            # Simulate a transient pinch-like dip mid-stroke due to motion blur.
            pinch = 0.25 if i in (3, 4) else 0.7
            lms = pointing_hand((x, 0.60), palm_len=palm_len, pinch_ratio=pinch)
            ev = self.fsm.update(lms, t)
            states_seen.add(ev.state)
            if ev.action:
                actions_seen.append(ev.action)

        # Continue for a few more frames right after the stroke, still with a
        # transiently low pinch_ratio, to cover the post-clear decay window. This is
        # NOT asserting pinch can never engage after a swipe (it legitimately can,
        # once the hand is truly stationary again) -- only that it can't engage in
        # the immediate wake of the stroke while speed is still decaying.
        for _ in range(3):
            t += dt
            lms = create_hand_landmarks(wrist_xy=(0.55, 0.60), palm_len=palm_len, pinch_ratio=0.25)
            ev = self.fsm.update(lms, t)
            states_seen.add(ev.state)
            if ev.action:
                actions_seen.append(ev.action)

        self.assertNotIn("PINCH_PENDING", states_seen, "Falsely entered PINCH_PENDING during/after swipe!")
        self.assertNotIn("CLICK", actions_seen, "False CLICK registered during/after swipe!")

    def test_edge_exit_clear_blocks_pinch_entry(self) -> None:
        """
        Right after palm_history is cleared by the near-edge guard, a hand presenting
        a pinch-eligible pose must not immediately enter PINCH_PENDING.
        """
        self.fsm.reset()
        t = 110.0
        dt = 1.0 / 30.0
        palm_len = 0.15

        # Drive the hand near the left edge to trigger the is_near_edge history clear.
        lms_edge = create_hand_landmarks(wrist_xy=(0.03, 0.60), palm_len=palm_len, pinch_ratio=0.7)
        ev = self.fsm.update(lms_edge, t)
        self.assertEqual(ev.state, "HOVER")

        # Immediately present a pinch-eligible, stationary-looking frame.
        t += dt
        lms_pinch = create_hand_landmarks(wrist_xy=(0.50, 0.60), palm_len=palm_len, pinch_ratio=0.10)
        ev = self.fsm.update(lms_pinch, t)
        self.assertNotEqual(ev.state, "PINCH_PENDING", "Pinch entry not blocked right after history clear!")

    def test_hand_exit_suppression(self) -> None:
        """
        A hand continuing down and out of frame may legitimately fire one
        SWIPE_DOWN partway through its travel -- that's correct now (see the
        scroll-down fix below), since a genuine scroll-down gesture and the
        start of a withdrawal motion look identical until the hand actually
        nears the edge. What must never happen is an action firing once the
        hand is actually within the near-edge withdrawal zone, or a second,
        spurious action as it crosses the edge.
        """
        self.fsm.reset()
        t = 70.0
        dt = 1.0 / 30.0
        palm_len = 0.15

        # Simulate user withdrawing their hand downwards toward desk/lap
        exit_events = []
        for i in range(7):
            t += dt
            # Hand drops from y=0.60 to y=0.95 (exiting frame at bottom)
            y = 0.60 + (0.35 * (i / 6.0))
            lms = pointing_hand((0.50, y), palm_len=palm_len)
            ev = self.fsm.update(lms, t)
            if ev.action:
                exit_events.append((ev.action, y))

        for action, y in exit_events:
            self.assertLess(
                y, 0.88,
                f"{action} fired while the hand was within the near-edge withdrawal zone (y={y:.2f})!",
            )
        self.assertLessEqual(len(exit_events), 1, f"Hand exit produced more than one action: {exit_events}")

    def test_scroll_down_registers_without_reaching_screen_edge(self) -> None:
        """
        Regression test for the reported "scroll down not working" bug: the
        boundary-exit guard used to reject SWIPE_DOWN whenever the palm ended up
        below y=0.68 (barely past mid-screen) or the wrist below y=0.76, even
        though the actual frame edge isn't until ~0.88. A natural downward
        scroll gesture routinely ends in that dead zone and was being wrongly
        treated as "withdrawing the hand," not an intentional scroll. It must
        now fire.
        """
        self.fsm.reset()
        t = 75.0
        dt = 1.0 / 30.0
        palm_len = 0.15

        events = []
        for i in range(4):
            t += dt
            y = 0.55 + 0.30 * (i / 3.0)  # ends at wrist y=0.85 (palm/m9 y=0.70) -- well
            lms = pointing_hand((0.50, y), palm_len=palm_len)  # short of the real edge (~0.88)
            ev = self.fsm.update(lms, t)
            if ev.action:
                events.append(ev.action)

        self.assertIn("SWIPE_DOWN", events, "Scroll-down failed to register despite staying clear of the screen edge!")


class _FakeClassification:
    def __init__(self, label: str, score: float) -> None:
        self.label = label
        self.score = score


class _FakeHandedness:
    def __init__(self, label: str, score: float) -> None:
        self.classification = [_FakeClassification(label, score)]


class _FakeHandLandmarks:
    def __init__(self, landmark) -> None:
        self.landmark = landmark


class TestSelectRightHand(unittest.TestCase):
    """Unit tests for main.select_right_hand (right-hand-only gesture control)."""

    def test_accepts_high_confidence_right(self) -> None:
        lms = create_hand_landmarks()
        result = select_right_hand(
            [_FakeHandLandmarks(lms)],
            [_FakeHandedness("Right", 0.95)],
        )
        self.assertIs(result, lms)

    def test_rejects_left(self) -> None:
        lms = create_hand_landmarks()
        result = select_right_hand(
            [_FakeHandLandmarks(lms)],
            [_FakeHandedness("Left", 0.95)],
        )
        self.assertIsNone(result)

    def test_rejects_low_confidence_right(self) -> None:
        lms = create_hand_landmarks()
        result = select_right_hand(
            [_FakeHandLandmarks(lms)],
            [_FakeHandedness("Right", 0.5)],
            min_score=0.7,
        )
        self.assertIsNone(result)

    def test_picks_right_among_multiple(self) -> None:
        left_lms = create_hand_landmarks(wrist_xy=(0.2, 0.5))
        right_lms = create_hand_landmarks(wrist_xy=(0.8, 0.5))
        result = select_right_hand(
            [_FakeHandLandmarks(left_lms), _FakeHandLandmarks(right_lms)],
            [_FakeHandedness("Left", 0.98), _FakeHandedness("Right", 0.97)],
        )
        self.assertIs(result, right_lms)

    def test_no_hands_returns_none(self) -> None:
        self.assertIsNone(select_right_hand([], []))


class TestRightHandGate(unittest.TestCase):
    """Unit tests for main.RightHandGate (hysteresis against handedness-classifier noise)."""

    def setUp(self) -> None:
        self.gate = RightHandGate(min_score=0.7, lock_grace_frames=12, relock_confirm_frames=3)
        self.lms = create_hand_landmarks()

    def _frame(self, label: str, score: float):
        return [_FakeHandLandmarks(self.lms)], [_FakeHandedness(label, score)]

    def test_locks_after_confirm_frames(self) -> None:
        for _ in range(self.gate.relock_confirm_frames - 1):
            landmarks, rejected = self.gate.update(*self._frame("Right", 0.95))
            self.assertIsNone(landmarks)
            self.assertFalse(rejected)
        landmarks, rejected = self.gate.update(*self._frame("Right", 0.95))
        self.assertIs(landmarks, self.lms)
        self.assertFalse(rejected)
        self.assertTrue(self.gate.locked)

    def test_survives_brief_misclassification_mid_gesture(self) -> None:
        for _ in range(self.gate.relock_confirm_frames):
            self.gate.update(*self._frame("Right", 0.95))
        self.assertTrue(self.gate.locked)

        # A handful of ambiguous/low-confidence frames (fewer than the grace period),
        # simulating the handedness classifier wobbling during a fast swipe/drag.
        for _ in range(self.gate.lock_grace_frames - 1):
            landmarks, rejected = self.gate.update(*self._frame("Right", 0.4))
            self.assertIsNotNone(landmarks, "Dropped a confirmed right hand on a brief wobble!")
            self.assertFalse(rejected, "Falsely flashed the wrong-hand banner during a brief wobble!")

        # Confidence recovers; should remain locked and tracking continuously.
        landmarks, rejected = self.gate.update(*self._frame("Right", 0.95))
        self.assertIs(landmarks, self.lms)
        self.assertFalse(rejected)

    def test_drops_after_sustained_left(self) -> None:
        for _ in range(self.gate.relock_confirm_frames):
            self.gate.update(*self._frame("Right", 0.95))
        self.assertTrue(self.gate.locked)

        landmarks, rejected = None, False
        for _ in range(self.gate.lock_grace_frames + 2):
            landmarks, rejected = self.gate.update(*self._frame("Left", 0.95))

        self.assertIsNone(landmarks)
        self.assertTrue(rejected, "Sustained left-hand presence should eventually be flagged!")
        self.assertFalse(self.gate.locked)

    def test_rejects_clear_left_before_lock(self) -> None:
        landmarks, rejected = self.gate.update(*self._frame("Left", 0.95))
        self.assertIsNone(landmarks)
        self.assertTrue(rejected)

    def test_no_hand_resets_gate(self) -> None:
        for _ in range(self.gate.relock_confirm_frames):
            self.gate.update(*self._frame("Right", 0.95))
        self.assertTrue(self.gate.locked)

        landmarks, rejected = self.gate.update([], [])
        self.assertIsNone(landmarks)
        self.assertFalse(rejected)
        self.assertFalse(self.gate.locked)


class TestIndexPointingSwipe(unittest.TestCase):
    """Unit tests for the (sole) index-finger-pointing swipe/scroll gesture.

    Open-palm swipe has been removed entirely -- pointing with the index finger
    (other fingers curled) is now the only gesture that can trigger a swipe/scroll,
    so that ambiguous hand motion (an open hand just moving around, e.g. while
    hovering for a click) is never mistaken for an intentional gesture.
    """

    def setUp(self) -> None:
        self.fsm = VirtualMirrorGestureFSM()

    def test_pointing_pose_detected(self) -> None:
        lms = pointing_hand((0.50, 0.60))
        self.assertTrue(self.fsm._is_pointing_pose(lms))

    def test_open_hand_not_pointing_pose(self) -> None:
        lms = create_hand_landmarks(wrist_xy=(0.50, 0.60), pinch_ratio=0.7)
        self.assertFalse(self.fsm._is_pointing_pose(lms))

    def test_index_point_swipe_registers(self) -> None:
        self.fsm.reset()
        t = 140.0
        dt = 1.0 / 30.0
        palm_len = 0.15

        events = []
        for i in range(7):
            t += dt
            x = 0.20 + (0.35 * (i / 6.0))
            lms = pointing_hand((x, 0.60), palm_len=palm_len)
            ev = self.fsm.update(lms, t)
            if ev.action:
                events.append(ev.action)

        self.assertIn("SWIPE_RIGHT", events, "Index-pointing swipe failed to register!")

    def test_open_hand_motion_never_swipes(self) -> None:
        """An open/ambiguous hand performing the exact same motion that would be
        a clean SWIPE_RIGHT while pointing must NOT trigger any swipe at all."""
        self.fsm.reset()
        t = 150.0
        dt = 1.0 / 30.0
        palm_len = 0.15

        events = []
        for i in range(7):
            t += dt
            x = 0.20 + (0.35 * (i / 6.0))
            lms = create_hand_landmarks(wrist_xy=(x, 0.60), palm_len=palm_len, pinch_ratio=0.7)
            ev = self.fsm.update(lms, t)
            if ev.action:
                events.append(ev.action)

        self.assertEqual(events, [], f"Ambiguous open-hand motion was treated as a gesture: {events}")

    def test_switching_pose_mid_stroke_does_not_falsely_fire(self) -> None:
        self.fsm.reset()
        t = 160.0
        dt = 1.0 / 30.0
        palm_len = 0.15

        events = []
        # 3 frames pointing, then 3 frames as an open (non-pointing) hand -- each
        # phase alone is one frame short of the 4-frame minimum _evaluate_swipe
        # requires, and switching out of the pointing pose drops the history, so
        # the two phases can't combine into a false-positive swipe.
        for i in range(3):
            t += dt
            x = 0.20 + (0.15 * (i / 2.0))
            lms = pointing_hand((x, 0.60), palm_len=palm_len)
            ev = self.fsm.update(lms, t)
            if ev.action:
                events.append(ev.action)

        for i in range(3):
            t += dt
            x = 0.35 + (0.20 * (i / 2.0))
            lms = create_hand_landmarks(wrist_xy=(x, 0.60), palm_len=palm_len, pinch_ratio=0.7)
            ev = self.fsm.update(lms, t)
            if ev.action:
                events.append(ev.action)

        self.assertEqual(events, [], f"Switching hand shape mid-stroke falsely fired a swipe: {events}")


if __name__ == "__main__":
    unittest.main()
