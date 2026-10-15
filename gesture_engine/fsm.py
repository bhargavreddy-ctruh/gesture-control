"""
Deterministic Kinematic Finite State Machine (FSM)
Zero-training gesture decoder with 3D anatomical pinch distance, palm orientation
visibility gates, and scale-invariant temporal swipe/scroll detection.
"""

import math
from collections import deque
from dataclasses import dataclass
from typing import Optional, Tuple, Deque, Any

from .filter import Point2DEuroFilter


@dataclass
class GestureEvent:
    """Represents a state machine transition or interaction event."""
    state: str                          # 'IDLE', 'HOVER', 'PINCH_PENDING', 'HOLD', 'DRAG'
    action: Optional[str]               # 'CLICK', 'HOLD_START', 'HOLD_END', 'DRAG_START', 'DRAG_MOVE', 'DRAG_END', 'SWIPE_*'
    cursor_xy: Tuple[float, float]      # Normalized (x, y) coordinates in [0.0, 1.0]
    pinch_ratio: float                  # Scale-invariant 3D pinch distance ratio
    palm_scale: float                   # Palm reference length in normalized coordinates
    dwell_progress: float = 0.0         # 0.0 to 1.0 progress toward HOLD activation
    current_speed: float = 0.0          # Current hand velocity (normalized units/s)
    current_disp: float = 0.0           # Current stroke displacement / palm_scale
    current_linearity: float = 0.0      # Current stroke straightness ratio
    palm_valid: bool = True             # True if palm is facing camera and fully visible


class VirtualMirrorGestureFSM:
    """
    Deterministic kinematic state machine for touchless virtual mirror interfaces.
    Uses 3D Euclidean distances, palm visibility gating, and Schmitt-trigger hysteresis.
    """

    def __init__(
        self,
        pinch_engage_thresh: float = 0.28,
        pinch_release_thresh: float = 0.38,
        deadzone_palm_ratio: float = 0.25,
        click_min_sec: float = 0.04,
        click_max_sec: float = 0.40,
        hold_trigger_sec: float = 0.45,
        swipe_window: int = 9,
        swipe_min_palm_disp: float = 0.90,
        swipe_min_screen_disp: float = 0.09,
        swipe_min_velocity: float = 0.38,
        swipe_linearity_min: float = 0.68,
        swipe_axis_ratio: float = 1.35,
        swipe_cooldown_sec: float = 0.50,
        swipe_min_palm_disp_fast: float = 0.45,
        swipe_linearity_min_fast: float = 0.55,
        swipe_fast_velocity: float = 1.0,
        swipe_speed_decay_tau: float = 0.15,
        pinch_entry_block_sec: float = 0.12,
        pinch_confirm_frames: int = 3,
    ) -> None:
        # Scale-invariant pinch thresholds (comfortable engage + clean hysteresis)
        self.pinch_engage_thresh = pinch_engage_thresh
        self.pinch_release_thresh = pinch_release_thresh

        # Spatial and temporal thresholds for Click, Hold, and Drag
        self.deadzone_palm_ratio = deadzone_palm_ratio
        self.click_min_sec = click_min_sec
        self.click_max_sec = click_max_sec
        self.hold_trigger_sec = hold_trigger_sec

        # Temporal swipe parameters (calibrated for deliberate gestures vs casual hand drift)
        self.swipe_window = swipe_window
        self.swipe_min_palm_disp = swipe_min_palm_disp
        self.swipe_min_screen_disp = swipe_min_screen_disp
        self.swipe_min_velocity = swipe_min_velocity
        self.swipe_linearity_min = swipe_linearity_min
        self.swipe_axis_ratio = swipe_axis_ratio
        self.swipe_cooldown_sec = swipe_cooldown_sec

        # Velocity-aware relaxation of the swipe displacement/linearity floors:
        # a short, fast flick can be unambiguously intentional (high velocity) without
        # covering as much physical distance as a slower, deliberate full-arm swipe.
        self.swipe_min_palm_disp_fast = swipe_min_palm_disp_fast
        self.swipe_linearity_min_fast = swipe_linearity_min_fast
        self.swipe_fast_velocity = swipe_fast_velocity

        # Decaying speed estimate + short post-clear cooldown so a hand that was moving
        # fast right as palm_history gets cleared (edge/boundary guard) isn't mistaken
        # for a stationary hand and allowed into PINCH_PENDING a frame later.
        self.swipe_speed_decay_tau = swipe_speed_decay_tau
        self.pinch_entry_block_sec = pinch_entry_block_sec

        # A pinch must be seen clearly (all geometric gates passing) for this many
        # consecutive frames before it's trusted enough to engage PINCH_PENDING,
        # so a single noisy depth/visibility frame can't be mistaken for a real pinch.
        self.pinch_confirm_frames = pinch_confirm_frames

        # Internal state variables
        self.state = "IDLE"
        self.pinch_start_time = 0.0
        self.anchor_xy: Tuple[float, float] = (0.5, 0.5)
        self.last_swipe_time = -100.0
        # Swipe/scroll motion history, tracked via the (filtered) index fingertip --
        # this is the sole swipe/scroll tracking point; it works whether the user
        # points with just their index finger or moves their whole open hand, since
        # the fingertip moves along with the hand either way.
        self.swipe_history: Deque[Tuple[float, float, float]] = deque(maxlen=self.swipe_window)
        self._smoothed_speed = 0.0
        self._last_update_time = 0.0
        self._last_history_clear_time = -100.0
        self._pinch_candidate_frames = 0

        # 1 Euro Filters for cursor stabilization
        self.thumb_filter = Point2DEuroFilter(min_cutoff=1.2, beta=0.35)
        self.index_filter = Point2DEuroFilter(min_cutoff=1.2, beta=0.35)
        self.palm_filter = Point2DEuroFilter(min_cutoff=1.5, beta=0.40)

    def reset(self) -> None:
        """Reset the FSM and all stabilization filters upon tracking loss."""
        self.state = "IDLE"
        self.pinch_start_time = 0.0
        self.swipe_history.clear()
        self.last_swipe_time = -100.0
        self._smoothed_speed = 0.0
        self._last_update_time = 0.0
        self._last_history_clear_time = -100.0
        self._pinch_candidate_frames = 0
        self.thumb_filter.reset()
        self.index_filter.reset()
        self.palm_filter.reset()

    def _clear_palm_history(self, now: float) -> None:
        """Clear the swipe/scroll motion history and record when it happened.

        Tracking this timestamp lets the pinch-entry gate briefly distrust a
        suddenly-empty history instead of reading it as a stationary hand.
        """
        self.swipe_history.clear()
        self._last_history_clear_time = now

    def _is_pointing_pose(self, landmarks: Any) -> bool:
        """True if the index finger is extended and at least two of the other
        three long fingers are curled -- a deliberate pointing pose.

        Swipe/scroll only recognizes this specific, intentional hand shape, not
        just any hand motion -- an open or ambiguous hand moving around (e.g.
        while hovering) must never be mistaken for a swipe gesture. Requiring
        only 2 of 3 other fingers curled (not all 3) keeps it forgiving for a
        kiosk setting, where pinky/ring curl inconsistently when people point.
        """
        w0 = landmarks[0]

        def dist(a: Any, b: Any) -> float:
            return math.hypot(a.x - b.x, a.y - b.y)

        index_extended = dist(w0, landmarks[8]) > dist(w0, landmarks[5]) * 1.15
        if not index_extended:
            return False
        curled = (
            dist(w0, landmarks[12]) < dist(w0, landmarks[9]) * 1.25,   # middle
            dist(w0, landmarks[16]) < dist(w0, landmarks[13]) * 1.25,  # ring
            dist(w0, landmarks[20]) < dist(w0, landmarks[17]) * 1.25,  # pinky
        )
        return sum(curled) >= 2

    def update(self, landmarks: Any, now: float) -> GestureEvent:
        """
        Process a single frame of MediaPipe hand landmarks with 3D geometric checks.
        """
        w0 = landmarks[0]   # Wrist
        t4 = landmarks[4]   # Thumb Tip
        i8 = landmarks[8]   # Index Tip
        m9 = landmarks[9]   # Middle MCP

        # Extract 3D normalized coordinates (MediaPipe provides .x, .y, .z)
        wz = getattr(w0, "z", 0.0)
        tz = getattr(t4, "z", 0.0)
        iz = getattr(i8, "z", 0.0)
        mz = getattr(m9, "z", 0.0)

        # Stabilize cursor coordinates with 1 Euro Filter
        tx, ty = self.thumb_filter.filter(t4.x, t4.y, now)
        ix, iy = self.index_filter.filter(i8.x, i8.y, now)

        # Scale-invariant palm length in 2D and 3D
        palm_scale_2d = math.hypot(m9.x - w0.x, m9.y - w0.y)
        palm_scale_3d = math.sqrt((m9.x - w0.x)**2 + (m9.y - w0.y)**2 + (mz - wz)**2)
        palm_scale = max(palm_scale_2d, 1e-4)

        # Cursor tracks the smoothed index fingertip directly (not a midpoint with the
        # thumb): this way, closing the thumb in for a pinch/click doesn't drag the
        # cursor off target -- the index tip stays put while the thumb comes to it.
        cursor_xy = (ix, iy)

        # 2D Euclidean pinch distance (jitter-free and scale-invariant)
        pinch_dist_2d = math.hypot(ix - tx, iy - ty)
        pinch_ratio = pinch_dist_2d / palm_scale

        # Depth disparity check between thumb tip and index tip
        # In a real physical pinch, both tips reside in roughly the same depth plane (ratio <= 0.35).
        # When thumb is tucked behind/under fingers in 2D projection, depth disparity is large (> 0.50).
        depth_disparity = abs(iz - tz) / palm_scale
        coplanar = depth_disparity <= 0.35

        # --- PALM VISIBILITY & BOUNDARY EXIT GATES ---
        is_distant = palm_scale < 0.07

        # 1. Edge boundary guard: Hand near edges has unstable tracking or is entering/exiting.
        border_x = 0.05
        border_y_top = 0.05
        border_y_bottom = 0.10  # Generous bottom margin where hands enter/leave desk & lap
        is_near_edge = (
            w0.x < border_x or w0.x > (1.0 - border_x) or
            m9.x < border_x or m9.x > (1.0 - border_x) or
            w0.y < border_y_top or w0.y > (1.0 - border_y_bottom) or
            m9.y < border_y_top or m9.y > (1.0 - border_y_bottom)
        )

        # 2. Minimum physical palm size: lowered from 0.045 to 0.015 so users can stand 2-4 meters away
        palm_size_ok = palm_scale >= 0.015

        # 3. Palm facing camera check (rejects steep downward/forward tilt / foreshortening)
        tilt_ratio = palm_scale_2d / max(1e-4, palm_scale_3d)
        min_tilt = 0.20 if is_distant else 0.32
        tilt_ok = tilt_ratio >= min_tilt

        # Pure palm visibility check:
        palm_valid = (not is_near_edge) and palm_size_ok and tilt_ok

        # 4. Pinch-specific geometric guards:
        # Distance-adaptive pinch thresholds
        active_engage_thresh = (
            min(0.36, self.pinch_engage_thresh + (0.07 - palm_scale) * 1.5)
            if is_distant else self.pinch_engage_thresh
        )
        self._active_release_thresh = (
            min(0.48, self.pinch_release_thresh + (0.07 - palm_scale) * 1.5)
            if is_distant else self.pinch_release_thresh
        )

        dist_t4_m9 = math.hypot(t4.x - m9.x, t4.y - m9.y)
        dist_t4_i8 = math.hypot(t4.x - i8.x, t4.y - i8.y)
        thumb_not_tucked = (dist_t4_i8 < dist_t4_m9 * 1.35)
        max_depth_disparity = 0.65 if is_distant else 0.38
        depth_coplanar = depth_disparity <= max_depth_disparity

        pinch_valid_to_engage = (
            palm_valid and
            (pinch_ratio < active_engage_thresh) and
            depth_coplanar and
            thumb_not_tucked
        )

        action: Optional[str] = None
        dwell_progress = 0.0

        # Only a deliberate pointing pose accumulates swipe/scroll history -- an open
        # or ambiguous hand moving around (e.g. while hovering for a click) must never
        # be mistaken for an intentional swipe.
        pointing = self._is_pointing_pose(landmarks)

        # If hand is right at the border, flush swipe history and briefly distrust
        # the pinch-speed gate (handled by _clear_palm_history). Simply not pointing
        # is the ordinary state for clicking/dragging, though, so just drop any
        # stale swipe history without touching that pinch-entry timer.
        if is_near_edge:
            self._clear_palm_history(now)
        elif not pointing:
            self.swipe_history.clear()
        else:
            # Raw (unfiltered) index-tip position, not the smoothed cursor coordinate --
            # the swipe gates (velocity/displacement/linearity) are tuned against raw
            # landmark motion, matching how the previous palm-tracked swipe worked;
            # feeding them the heavily-smoothed cursor position would add lag and
            # under-report fast, short flicks.
            self.swipe_history.append((i8.x, i8.y, now))

        # Calculate live motion diagnostics (pure 1:1 physical displacement)
        current_speed = 0.0
        current_disp = 0.0
        current_linearity = 0.0

        if len(self.swipe_history) >= 2:
            x0, y0, t0 = self.swipe_history[0]
            x1, y1, t1 = self.swipe_history[-1]
            dt_diag = max(1e-4, t1 - t0)
            net_d = math.hypot(x1 - x0, y1 - y0)
            current_speed = net_d / dt_diag
            current_disp = net_d / palm_scale

            path_sum = sum(
                math.hypot(self.swipe_history[k][0] - self.swipe_history[k-1][0],
                           self.swipe_history[k][1] - self.swipe_history[k-1][1])
                for k in range(1, len(self.swipe_history))
            )
            current_linearity = (net_d / path_sum) if path_sum > 1e-4 else 0.0

        # Decaying speed estimate: holds the last real reading across a history
        # clear instead of collapsing to 0.0 the instant the deque empties, so a hand
        # that was genuinely moving fast isn't mistaken for stationary a frame later.
        dt_frame = (now - self._last_update_time) if self._last_update_time > 0.0 else (1.0 / 30.0)
        dt_frame = max(1e-4, dt_frame)
        self._last_update_time = now
        decay = math.exp(-dt_frame / self.swipe_speed_decay_tau)
        if len(self.swipe_history) >= 2:
            self._smoothed_speed = max(current_speed, self._smoothed_speed * decay)
        else:
            self._smoothed_speed *= decay

        # --- STATE MACHINE TRANSITIONS ---
        if self.state in ("IDLE", "HOVER"):
            self.state = "HOVER"

            # 1. Check for active directional swipe / scroll first (pointing pose only)
            action = self._evaluate_swipe(
                palm_scale=palm_scale,
                now=now,
                curr_m9=(m9.x, m9.y),
                curr_w0=(w0.x, w0.y),
                is_near_edge=is_near_edge,
            ) if pointing else None

            # 2. Only allow pinch engagement if palm is fully visible, co-planar, and hand not
            #    moving fast -- and only once that's been true for several consecutive frames,
            #    so a single noisy/ambiguous-depth frame can't be mistaken for a real pinch.
            if action is None and pinch_valid_to_engage:
                recently_cleared = (now - self._last_history_clear_time) < self.pinch_entry_block_sec
                if self._smoothed_speed < (self.swipe_min_velocity * 0.75) and not recently_cleared:
                    self._pinch_candidate_frames += 1
                    if self._pinch_candidate_frames >= self.pinch_confirm_frames:
                        self.state = "PINCH_PENDING"
                        self.pinch_start_time = now
                        self.anchor_xy = cursor_xy
                else:
                    self._pinch_candidate_frames = 0
            else:
                self._pinch_candidate_frames = 0

        elif self.state == "PINCH_PENDING":
            elapsed = now - self.pinch_start_time
            disp_ratio = math.hypot(
                cursor_xy[0] - self.anchor_xy[0],
                cursor_xy[1] - self.anchor_xy[1]
            ) / palm_scale

            dwell_progress = min(1.0, elapsed / self.hold_trigger_sec)

            # 1. Check for pinch release first (opening fingers after a valid pinch engagement)
            release_thresh = getattr(self, "_active_release_thresh", self.pinch_release_thresh)
            if pinch_ratio > release_thresh:
                if self.click_min_sec <= elapsed < self.click_max_sec and disp_ratio < self.deadzone_palm_ratio:
                    action = "CLICK"
                self.state = "HOVER"
                dwell_progress = 0.0
            # 2. If hand leaves frame or becomes invalid during pinch, cancel without click
            elif is_near_edge or (not palm_size_ok):
                self.state = "HOVER"
                dwell_progress = 0.0
            elif disp_ratio >= self.deadzone_palm_ratio:
                # Moved outside stationary deadzone while pinched -> DRAG
                self.state = "DRAG"
                action = "DRAG_START"
                dwell_progress = 0.0
            elif elapsed >= self.hold_trigger_sec:
                # Held stationary inside deadzone beyond dwell duration -> HOLD
                self.state = "HOLD"
                action = "HOLD_START"
                dwell_progress = 1.0

        elif self.state == "HOLD":
            disp_ratio = math.hypot(
                cursor_xy[0] - self.anchor_xy[0],
                cursor_xy[1] - self.anchor_xy[1]
            ) / palm_scale

            dwell_progress = 1.0

            release_thresh = getattr(self, "_active_release_thresh", self.pinch_release_thresh)
            if pinch_ratio > release_thresh or not palm_valid:
                self.state = "HOVER"
                action = "HOLD_END"
                dwell_progress = 0.0
            elif disp_ratio >= self.deadzone_palm_ratio:
                # Seamless transition from Hold directly into Drag
                self.state = "DRAG"
                action = "DRAG_START"
                dwell_progress = 0.0

        elif self.state == "DRAG":
            # SWIPE ESCAPE: If user performs a fast swipe stroke while in DRAG, break out immediately!
            swipe_action = self._evaluate_swipe(
                palm_scale=palm_scale,
                now=now,
                curr_m9=(m9.x, m9.y),
                curr_w0=(w0.x, w0.y),
                is_near_edge=is_near_edge,
            ) if pointing else None
            if swipe_action is not None:
                self.state = "HOVER"
                action = swipe_action
            elif pinch_ratio > self.pinch_release_thresh or not palm_valid:
                self.state = "HOVER"
                action = "DRAG_END"
            else:
                action = "DRAG_MOVE"

        return GestureEvent(
            state=self.state,
            action=action,
            cursor_xy=cursor_xy,
            pinch_ratio=pinch_ratio,
            palm_scale=palm_scale,
            dwell_progress=dwell_progress,
            current_speed=current_speed,
            current_disp=current_disp,
            current_linearity=current_linearity,
            palm_valid=palm_valid,
        )

    def _evaluate_swipe(
        self,
        palm_scale: float,
        now: float,
        curr_m9: Optional[Tuple[float, float]] = None,
        curr_w0: Optional[Tuple[float, float]] = None,
        is_near_edge: bool = False,
    ) -> Optional[str]:
        """
        Evaluate candidate swipe/scroll gesture across multi-frame sliding intervals
        of the index-fingertip motion history.
        Includes boundary exit guards to eliminate false scroll-down upon withdrawing hand.
        """
        if is_near_edge:
            self._clear_palm_history(now)
            return None

        history = self.swipe_history
        if len(history) < 4:
            return None
        if (now - self.last_swipe_time) < self.swipe_cooldown_sec:
            return None

        x1, y1, t1 = history[-1]
        best_candidate: Optional[str] = None
        max_strength = 0.0

        # Resolve current reference points for exit boundary detection
        m9_x, m9_y = curr_m9 if curr_m9 is not None else (x1, y1)
        w0_x = curr_w0[0] if curr_w0 is not None else m9_x
        w0_y = curr_w0[1] if curr_w0 is not None else (m9_y + palm_scale)

        min_start = max(0, len(history) - self.swipe_window)
        max_start = len(history) - 3

        for start_idx in range(min_start, max_start):
            x0, y0, t0 = history[start_idx]
            dt = t1 - t0
            if dt <= 1e-4:
                continue

            dx = x1 - x0
            dy = y1 - y0

            # 1:1 physical displacement in normalized frame coordinates (no artificial vertical bias)
            net_disp = math.hypot(dx, dy)

            # Gate 1: Scale-invariant and absolute displacement & velocity
            min_screen_disp = min(self.swipe_min_screen_disp, palm_scale * 0.70)
            if net_disp < min_screen_disp:
                continue
            velocity = net_disp / dt
            min_velocity = min(self.swipe_min_velocity, palm_scale * 3.8)
            if velocity < min_velocity:
                continue

            # Velocity-aware displacement/linearity floors
            fast_velocity = min(self.swipe_fast_velocity, min_velocity * 2.5)
            frac = (velocity - min_velocity) / max(
                1e-4, fast_velocity - min_velocity
            )
            frac = min(1.0, max(0.0, frac))
            required_disp = self.swipe_min_palm_disp - frac * (
                self.swipe_min_palm_disp - self.swipe_min_palm_disp_fast
            )
            required_linearity = self.swipe_linearity_min - frac * (
                self.swipe_linearity_min - self.swipe_linearity_min_fast
            )

            if (net_disp / palm_scale) < required_disp:
                continue

            # Gate 2: Path linearity (straightness) ratio
            path_length = 0.0
            for k in range(start_idx + 1, len(history)):
                xa, ya, _ = history[k - 1]
                xb, yb, _ = history[k]
                path_length += math.hypot(xb - xa, yb - ya)

            if path_length <= 1e-5 or (net_disp / path_length) < required_linearity:
                continue

            # Gate 3: Dominant axis disambiguation
            swipe_cmd: Optional[str] = None
            if abs(dx) > self.swipe_axis_ratio * abs(dy):
                swipe_cmd = "SWIPE_RIGHT" if dx > 0 else "SWIPE_LEFT"
            elif abs(dy) > self.swipe_axis_ratio * abs(dx):
                swipe_cmd = "SWIPE_DOWN" if dy > 0 else "SWIPE_UP"

            if swipe_cmd is None:
                continue

            # Gate 4: BOUNDARY EXIT GUARDS
            # If the motion trajectory ends right at a screen border in the direction of
            # movement, it is the user withdrawing/taking their hand out of the camera
            # view, NOT an intentional command! Judged from the wrist/palm (m9, w0) only,
            # matching is_near_edge's actual border (~0.06-0.12 margin): these thresholds
            # used to sit far short of that (e.g. 0.68 for SWIPE_DOWN, barely past
            # mid-screen), which wrongly cancelled ordinary downward/upward scroll
            # gestures long before the hand was anywhere near actually leaving frame.
            if swipe_cmd == "SWIPE_DOWN":
                # Withdrawing hand downwards toward desk/lap
                if m9_y > 0.84 or w0_y > 0.88:
                    self._clear_palm_history(now)
                    return None
            elif swipe_cmd == "SWIPE_UP":
                # Moving hand upwards off top edge
                if m9_y < 0.09 or w0_y < 0.07:
                    self._clear_palm_history(now)
                    return None
            elif swipe_cmd == "SWIPE_LEFT":
                # Moving hand off left edge
                if m9_x < 0.09 or w0_x < 0.07:
                    self._clear_palm_history(now)
                    return None
            elif swipe_cmd == "SWIPE_RIGHT":
                # Moving hand off right edge
                if m9_x > 0.91 or w0_x > 0.93:
                    self._clear_palm_history(now)
                    return None

            strength = (net_disp / palm_scale) * velocity
            if strength > max_strength:
                max_strength = strength
                best_candidate = swipe_cmd

        if best_candidate is not None:
            self.last_swipe_time = now
            self._clear_palm_history(now)
            return best_candidate

        return None
