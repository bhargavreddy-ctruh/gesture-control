"""
Virtual Try-On Smart Mirror Visualizer
Provides modern UI overlay, hand skeleton rendering, animated dwell progress,
real-time swipe motion diagnostics, interactive tutorial coach, and virtual garment carousel.
"""

import math
import time
from typing import List, Tuple, Optional, Any, Dict
import cv2
import numpy as np

from .fsm import GestureEvent


class GestureVisualizer:
    """Renders virtual try-on smart mirror HUD, cursor, and interactive elements."""

    HAND_CONNECTIONS = [
        (0, 1), (1, 2), (2, 3), (3, 4),           # Thumb
        (0, 5), (5, 6), (6, 7), (7, 8),           # Index
        (5, 9), (9, 10), (10, 11), (11, 12),       # Middle
        (9, 13), (13, 14), (14, 15), (15, 16),    # Ring
        (13, 17), (17, 18), (18, 19), (19, 20),   # Pinky
        (0, 17)                                   # Base
    ]

    def __init__(self, width: int = 640, height: int = 480) -> None:
        self.width = width
        self.height = height

        # Active gesture banner tracking
        self.active_banner = "NONE"
        self.banner_timestamp = 0.0
        self.banner_color = (180, 180, 180)

        # Interactive Virtual Mirror Mock State
        self.garment_index = 0
        self.garments = [
            {"name": "Cashmere Overshirt", "color": "Navy", "price": "$189"},
            {"name": "Silk Resort Blazer", "color": "Olive", "price": "$245"},
            {"name": "Tailored Linen Coat", "color": "Beige", "price": "$210"},
            {"name": "Merino Wool Knit", "color": "Charcoal", "price": "$165"},
        ]
        self.categories = ["Tops", "Outerwear", "Bottoms", "Accessories"]
        self.category_index = 0
        self.zoom_level = 1.0
        self.model_rotation = 0.0
        self.last_cursor_xy: Optional[Tuple[float, float]] = None

        # Tutorial / Learning Coach state
        self.tutorial_active = False
        self.tutorial_lesson: Optional[Dict[str, Any]] = None
        self.show_diagnostics = True

    def trigger_action(self, action: str, cursor_xy: Tuple[float, float]) -> None:
        """Update virtual mirror mock state in response to gestures."""
        self.active_banner = action
        self.banner_timestamp = time.perf_counter()

        if action == "SWIPE_RIGHT":
            self.garment_index = (self.garment_index + 1) % len(self.garments)
            self.banner_color = (0, 220, 255)
        elif action == "SWIPE_LEFT":
            self.garment_index = (self.garment_index - 1) % len(self.garments)
            self.banner_color = (0, 220, 255)
        elif action == "SWIPE_DOWN":
            self.category_index = (self.category_index + 1) % len(self.categories)
            self.banner_color = (255, 160, 0)
        elif action == "SWIPE_UP":
            self.category_index = (self.category_index - 1) % len(self.categories)
            self.banner_color = (255, 160, 0)
        elif action == "CLICK":
            self.banner_color = (50, 255, 50)
        elif action == "HOLD_START":
            self.zoom_level = 1.35 if self.zoom_level == 1.0 else 1.0
            self.banner_color = (255, 100, 255)
        elif action == "DRAG_MOVE":
            if self.last_cursor_xy is not None:
                dx = cursor_xy[0] - self.last_cursor_xy[0]
                self.model_rotation = (self.model_rotation + dx * 360.0) % 360.0
            self.banner_color = (50, 255, 150)

        self.last_cursor_xy = cursor_xy

    def set_tutorial_lesson(self, lesson: Optional[Dict[str, Any]]) -> None:
        """Set active interactive learning tutorial step."""
        self.tutorial_lesson = lesson
        self.tutorial_active = lesson is not None

    def render(
        self,
        frame: np.ndarray,
        landmarks: Optional[Any],
        event: Optional[GestureEvent],
        fps: float = 0.0,
        hand_rejected: bool = False,
    ) -> np.ndarray:
        """Render virtual mirror HUD, interactive carousel, diagnostics, and cursor."""
        h, w = frame.shape[:2]
        overlay = frame.copy()

        # 1. Top status bar
        self._render_header(overlay, fps)

        # 2. Interactive Garment Carousel & Category Card
        self._render_mirror_kiosk(overlay)

        # 3. Motion Diagnostics HUD (Real-Time Speed, Distance, Linearity)
        if self.show_diagnostics and event is not None:
            self._render_motion_diagnostics(overlay, event)

        # 4. Hand Skeleton
        if landmarks is not None:
            self._render_skeleton(overlay, landmarks, w, h)

        # 4b. Wrong-hand notice (a hand is visible but isn't the controlling right hand)
        if hand_rejected:
            self._render_wrong_hand_notice(overlay, w, h)

        # 5. Cursor & Interaction Feedback
        if event is not None and event.state != "IDLE":
            self._render_cursor(overlay, event, w, h)

        # 6. Tutorial Learning Guide & Directional Arrows
        if self.tutorial_active and self.tutorial_lesson is not None:
            self._render_tutorial_guide(overlay, w, h)

        # 7. Notification Banner for Triggered Gestures
        self._render_banner(overlay, w, h)

        cv2.addWeighted(overlay, 0.95, frame, 0.05, 0, frame)
        return frame

    def _render_wrong_hand_notice(self, img: np.ndarray, w: int, h: int) -> None:
        """Warn that a detected hand is being ignored because it isn't the right hand."""
        text = "RIGHT HAND ONLY"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, 0.6, 2)
        x = (w - tw) // 2
        y = 80
        cv2.rectangle(img, (x - 14, y - th - 12), (x + tw + 14, y + 10), (20, 20, 30), -1)
        cv2.rectangle(img, (x - 14, y - th - 12), (x + tw + 14, y + 10), (0, 140, 255), 1)
        cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 170, 255), 2)

    def _render_header(self, img: np.ndarray, fps: float) -> None:
        """Render top status bar."""
        cv2.rectangle(img, (0, 0), (self.width, 42), (18, 18, 22), -1)
        cv2.line(img, (0, 42), (self.width, 42), (55, 55, 65), 1)

        title = "SMART MIRROR VIRTUAL TRY-ON"
        cv2.putText(img, title, (16, 26), cv2.FONT_HERSHEY_DUPLEX, 0.52, (240, 240, 240), 1)

        mode_badge = "[PRACTICE COACH: 't']" if self.tutorial_active else "[KEY HINTS: 't'=Coach, 's'=Demo, 'r'=Reset]"
        cv2.putText(img, mode_badge, (self.width // 2 - 120, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 220, 255), 1)

        fps_text = f"FPS: {fps:.1f}"
        cv2.putText(img, fps_text, (self.width - 85, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (160, 200, 160), 1)

    def _render_mirror_kiosk(self, img: np.ndarray) -> None:
        """Render garment selection carousel at bottom."""
        card_h = 76
        card_y = self.height - card_h - 10
        card_w = self.width - 24
        card_x = 12

        sub_img = img[card_y:card_y + card_h, card_x:card_x + card_w]
        white_rect = np.full(sub_img.shape, 25, dtype=np.uint8)
        res = cv2.addWeighted(sub_img, 0.30, white_rect, 0.70, 0)
        img[card_y:card_y + card_h, card_x:card_x + card_w] = res
        cv2.rectangle(img, (card_x, card_y), (card_x + card_w, card_y + card_h), (80, 80, 95), 1)

        # Category Scroll Bar
        cat_active = self.categories[self.category_index].upper()
        cat_str = f"CATEGORY: [{cat_active}]  (Swipe/Scroll UP/DOWN)"
        cv2.putText(img, cat_str, (card_x + 16, card_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 220, 255), 1)

        # Garment Name & Price
        current_g = self.garments[self.garment_index]
        name_str = f"{current_g['name']}  •  {current_g['color']}  ({current_g['price']})"
        cv2.putText(img, name_str, (card_x + 16, card_y + 44), cv2.FONT_HERSHEY_DUPLEX, 0.55, (255, 255, 255), 1)

        # Interaction Hints
        hint_str = f"Item {self.garment_index + 1}/{len(self.garments)} | Rot: {int(self.model_rotation)}° | Zoom: {self.zoom_level:.1f}x | Swipe L/R: Items | Swipe U/D: Categories"
        cv2.putText(img, hint_str, (card_x + 16, card_y + 65), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (170, 175, 185), 1)

    def _render_motion_diagnostics(self, img: np.ndarray, event: GestureEvent) -> None:
        """Render real-time swipe/motion diagnostic meters."""
        box_w = 180
        box_h = 92
        box_x = self.width - box_w - 12
        box_y = 50

        # Translucent background
        sub = img[box_y:box_y + box_h, box_x:box_x + box_w]
        bg = np.full(sub.shape, 20, dtype=np.uint8)
        img[box_y:box_y + box_h, box_x:box_x + box_w] = cv2.addWeighted(sub, 0.25, bg, 0.75, 0)
        cv2.rectangle(img, (box_x, box_y), (box_x + box_w, box_y + box_h), (60, 60, 75), 1)

        # Header
        cv2.putText(img, "GESTURE METERS", (box_x + 8, box_y + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 180, 180), 1)

        # Speed bar (threshold: 0.38)
        spd_ready = event.current_speed >= 0.38
        spd_col = (0, 255, 100) if spd_ready else (100, 150, 255)
        spd_pct = min(1.0, event.current_speed / 0.80)
        cv2.putText(img, f"Speed: {event.current_speed:.2f}", (box_x + 8, box_y + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.32, spd_col, 1)
        cv2.rectangle(img, (box_x + 90, box_y + 21), (box_x + 90 + int(75 * spd_pct), box_y + 27), spd_col, -1)

        # Stroke distance bar (threshold: 0.90x palm)
        disp_ready = event.current_disp >= 0.90
        disp_col = (0, 255, 100) if disp_ready else (100, 150, 255)
        disp_pct = min(1.0, event.current_disp / 1.50)
        cv2.putText(img, f"Dist:  {event.current_disp:.2f}x", (box_x + 8, box_y + 42), cv2.FONT_HERSHEY_SIMPLEX, 0.32, disp_col, 1)
        cv2.rectangle(img, (box_x + 90, box_y + 35), (box_x + 90 + int(75 * disp_pct), box_y + 41), disp_col, -1)

        # Pinch Ratio bar (threshold: 0.28)
        pinch_active = event.pinch_ratio <= 0.28
        pinch_col = (0, 255, 100) if pinch_active else (200, 200, 200)
        cv2.putText(img, f"Pinch: {event.pinch_ratio:.2f}", (box_x + 8, box_y + 56), cv2.FONT_HERSHEY_SIMPLEX, 0.32, pinch_col, 1)

        # Palm Visibility Gate Status
        if event.palm_valid:
            vis_text = "Palm: VISIBLE"
            vis_col = (0, 255, 100)
        else:
            vis_text = "Palm: BLOCKED/TUCKED"
            vis_col = (0, 140, 255)
        cv2.putText(img, vis_text, (box_x + 8, box_y + 70), cv2.FONT_HERSHEY_SIMPLEX, 0.31, vis_col, 1)

        # State / Posture Status
        posture = "OPEN PALM" if event.pinch_ratio >= 0.28 else "PINCHING"
        post_col = (0, 255, 100) if event.pinch_ratio >= 0.28 else (0, 220, 255)
        cv2.putText(img, f"State: {event.state}", (box_x + 8, box_y + 84), cv2.FONT_HERSHEY_SIMPLEX, 0.32, post_col, 1)

    def _render_tutorial_guide(self, img: np.ndarray, w: int, h: int) -> None:
        """Render animated instructional guide card and directional arrow."""
        if self.tutorial_lesson is None:
            return

        title = self.tutorial_lesson.get("title", "")
        tip = self.tutorial_lesson.get("tip", "")
        direction = self.tutorial_lesson.get("direction", "")

        # Top Center Card
        card_w = 420
        card_h = 58
        card_x = (w - card_w) // 2
        card_y = 52

        sub = img[card_y:card_y + card_h, card_x:card_x + card_w]
        bg = np.full(sub.shape, 22, dtype=np.uint8)
        img[card_y:card_y + card_h, card_x:card_x + card_w] = cv2.addWeighted(sub, 0.20, bg, 0.80, 0)
        cv2.rectangle(img, (card_x, card_y), (card_x + card_w, card_y + card_h), (0, 220, 255), 2)

        cv2.putText(img, title, (card_x + 14, card_y + 22), cv2.FONT_HERSHEY_DUPLEX, 0.52, (0, 255, 255), 1)
        cv2.putText(img, tip, (card_x + 14, card_y + 44), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (220, 225, 230), 1)

        # Draw Animated Motion Arrow / Guide at center of screen
        cx, cy = w // 2, h // 2 - 25
        t = time.perf_counter()
        pulse = (math.sin(t * 6.0) + 1.0) * 0.5  # 0.0 to 1.0

        arrow_color = (0, 230, 255)
        if direction == "RIGHT":
            # Animated arrow pointing right
            p_start = (cx - 90, cy)
            p_end = (cx + 90, cy)
            cv2.arrowedLine(img, p_start, p_end, arrow_color, 4, tipLength=0.25)
            # Moving dot along arrow
            dot_x = int(p_start[0] + (p_end[0] - p_start[0]) * ((t * 2.0) % 1.0))
            cv2.circle(img, (dot_x, cy), 7, (255, 255, 255), -1)
        elif direction == "LEFT":
            # Animated arrow pointing left
            p_start = (cx + 90, cy)
            p_end = (cx - 90, cy)
            cv2.arrowedLine(img, p_start, p_end, arrow_color, 4, tipLength=0.25)
            dot_x = int(p_start[0] + (p_end[0] - p_start[0]) * ((t * 2.0) % 1.0))
            cv2.circle(img, (dot_x, cy), 7, (255, 255, 255), -1)
        elif direction == "DOWN":
            # Animated arrow pointing down
            p_start = (cx, cy - 70)
            p_end = (cx, cy + 70)
            cv2.arrowedLine(img, p_start, p_end, (255, 170, 0), 4, tipLength=0.25)
            dot_y = int(p_start[1] + (p_end[1] - p_start[1]) * ((t * 2.0) % 1.0))
            cv2.circle(img, (cx, dot_y), 7, (255, 255, 255), -1)
        elif direction == "UP":
            # Animated arrow pointing up
            p_start = (cx, cy + 70)
            p_end = (cx, cy - 70)
            cv2.arrowedLine(img, p_start, p_end, (255, 170, 0), 4, tipLength=0.25)
            dot_y = int(p_start[1] + (p_end[1] - p_start[1]) * ((t * 2.0) % 1.0))
            cv2.circle(img, (cx, dot_y), 7, (255, 255, 255), -1)
        elif direction == "CLICK":
            radius = int(18 + pulse * 12)
            cv2.circle(img, (cx, cy), radius, (50, 255, 50), 2)
            cv2.putText(img, "PINCH & RELEASE", (cx - 65, cy + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (50, 255, 50), 1)
        elif direction == "HOLD":
            angle = int(pulse * 360)
            cv2.ellipse(img, (cx, cy), (28, 28), -90, 0, angle, (255, 80, 255), 4)
            cv2.putText(img, "STATIONARY HOLD", (cx - 70, cy + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 80, 255), 1)
        elif direction == "DRAG":
            p_start = (cx - 70, cy)
            p_end = (cx + 70, cy)
            cv2.line(img, p_start, p_end, (50, 255, 150), 3)
            cv2.circle(img, (int(cx - 50 + pulse * 100), cy), 8, (50, 255, 150), -1)
            cv2.putText(img, "PINCH & SLIDE", (cx - 55, cy + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (50, 255, 150), 1)

    def _render_skeleton(self, img: np.ndarray, landmarks: Any, w: int, h: int) -> None:
        """Render clean skeletal wireframe."""
        points: List[Tuple[int, int]] = []
        for lm in landmarks:
            px, py = int(lm.x * w), int(lm.y * h)
            points.append((px, py))

        for p1, p2 in self.HAND_CONNECTIONS:
            if p1 < len(points) and p2 < len(points):
                cv2.line(img, points[p1], points[p2], (70, 70, 90), 1)

        for i, pt in enumerate(points):
            radius = 3 if i not in (4, 8, 9) else 4
            color = (120, 180, 255) if i in (4, 8) else (90, 100, 120)
            cv2.circle(img, pt, radius, color, -1)

    def _render_cursor(self, img: np.ndarray, event: GestureEvent, w: int, h: int) -> None:
        """Render high-visibility stabilized cursor with stateful color & dwell progress."""
        cx, cy = int(event.cursor_xy[0] * w), int(event.cursor_xy[1] * h)

        if event.state == "HOVER":
            cursor_color = (255, 200, 0)
            ring_radius = 12
        elif event.state == "PINCH_PENDING":
            cursor_color = (0, 230, 255)
            ring_radius = 14
        elif event.state == "HOLD":
            cursor_color = (255, 80, 255)
            ring_radius = 18
        elif event.state == "DRAG":
            cursor_color = (50, 255, 50)
            ring_radius = 16
        else:
            cursor_color = (200, 200, 200)
            ring_radius = 10

        cv2.circle(img, (cx, cy), ring_radius + 4, (cursor_color[0] // 3, cursor_color[1] // 3, cursor_color[2] // 3), 2)
        cv2.circle(img, (cx, cy), ring_radius, cursor_color, 2)
        cv2.circle(img, (cx, cy), 4, cursor_color, -1)

        if event.state == "PINCH_PENDING" and event.dwell_progress > 0.05:
            angle = int(event.dwell_progress * 360)
            cv2.ellipse(
                img,
                (cx, cy),
                (ring_radius + 6, ring_radius + 6),
                -90,
                0,
                angle,
                (0, 255, 255),
                3,
            )

        badge = f"{event.state}"
        cv2.putText(
            img,
            badge,
            (cx + 18, cy + 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.40,
            cursor_color,
            1,
        )

    def _render_banner(self, img: np.ndarray, w: int, h: int) -> None:
        """Render animated notification when a gesture triggers."""
        now = time.perf_counter()
        elapsed = now - self.banner_timestamp

        if elapsed > 1.2 or self.active_banner == "NONE":
            return

        text = f"GESTURE: {self.active_banner}"
        box_w = 340
        box_h = 42
        box_x = (w - box_w) // 2
        box_y = 114 if self.tutorial_active else 54

        sub = img[box_y:box_y + box_h, box_x:box_x + box_w]
        color_block = np.full(sub.shape, 30, dtype=np.uint8)
        img[box_y:box_y + box_h, box_x:box_x + box_w] = cv2.addWeighted(sub, 0.2, color_block, 0.8, 0)
        cv2.rectangle(img, (box_x, box_y), (box_x + box_w, box_y + box_h), self.banner_color, 2)

        cv2.putText(
            img,
            text,
            (box_x + 18, box_y + 27),
            cv2.FONT_HERSHEY_DUPLEX,
            0.62,
            self.banner_color,
            2,
        )
