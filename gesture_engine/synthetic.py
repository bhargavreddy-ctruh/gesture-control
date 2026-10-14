"""
Synthetic Hand Landmarks Generator
Used for automated unit testing, algorithm verification, and interactive demos
without requiring physical camera access or manual gesturing.
"""

import math
from typing import List, Tuple


class LandmarkMock:
    """Mock MediaPipe Normalized Landmark with x, y, z attributes."""
    __slots__ = ("x", "y", "z")

    def __init__(self, x: float, y: float, z: float = 0.0) -> None:
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)

    def __repr__(self) -> str:
        return f"Landmark(x={self.x:.3f}, y={self.y:.3f}, z={self.z:.3f})"


def create_hand_landmarks(
    wrist_xy: Tuple[float, float] = (0.5, 0.7),
    palm_len: float = 0.20,
    pinch_ratio: float = 0.60,
    hand_angle_rad: float = 0.0,
) -> List[LandmarkMock]:
    """
    Generate an anatomically consistent 21-landmark mock hand.

    Parameters:
        wrist_xy: (x, y) coordinates of Landmark 0 (Wrist).
        palm_len: Distance from Wrist (0) to Middle MCP (9) in normalized units.
        pinch_ratio: Ratio of distance(Thumb Tip, Index Tip) to palm_len.
                     < 0.32 simulates pinch engaged.
                     > 0.45 simulates open pinch.
        hand_angle_rad: Rotation of the hand in radians (0 is pointing up).

    Returns:
        List of 21 LandmarkMock objects.
    """
    wx, wy = wrist_xy
    cos_a = math.cos(hand_angle_rad)
    sin_a = math.sin(hand_angle_rad)

    def rot(dx: float, dy: float) -> Tuple[float, float]:
        return (wx + dx * cos_a - dy * sin_a, wy + dx * sin_a + dy * cos_a)

    # Landmark 0: Wrist
    w0 = LandmarkMock(wx, wy, 0.0)

    # Middle MCP (Landmark 9) is positioned palm_len directly forward
    m9_x, m9_y = rot(0.0, -palm_len)
    m9 = LandmarkMock(m9_x, m9_y, 0.0)

    # Thumb CMC (1), MCP (2), IP (3), TIP (4)
    # When pinch_ratio decreases, Thumb Tip (4) and Index Tip (8) converge.
    pinch_dist = pinch_ratio * palm_len
    cursor_x, cursor_y = rot(0.0, -palm_len * 1.35)

    # Landmark 4: Thumb Tip
    t4 = LandmarkMock(cursor_x - pinch_dist * 0.5, cursor_y, 0.0)
    # Landmark 8: Index Tip
    i8 = LandmarkMock(cursor_x + pinch_dist * 0.5, cursor_y, 0.0)

    # Fill full 21 landmarks for MediaPipe format compatibility
    lms: List[LandmarkMock] = [LandmarkMock(0, 0, 0) for _ in range(21)]
    lms[0] = w0
    lms[9] = m9
    lms[4] = t4
    lms[8] = i8

    # Proximal & intermediate joints
    t1_x, t1_y = rot(-palm_len * 0.25, -palm_len * 0.3)
    t2_x, t2_y = rot(-palm_len * 0.45, -palm_len * 0.6)
    t3_x, t3_y = rot(-palm_len * 0.35, -palm_len * 0.9)
    lms[1] = LandmarkMock(t1_x, t1_y, 0.0)
    lms[2] = LandmarkMock(t2_x, t2_y, 0.0)
    lms[3] = LandmarkMock(t3_x, t3_y, 0.0)

    i5_x, i5_y = rot(-palm_len * 0.20, -palm_len * 0.9)
    i6_x, i6_y = rot(-palm_len * 0.15, -palm_len * 1.1)
    i7_x, i7_y = rot(-palm_len * 0.10, -palm_len * 1.25)
    lms[5] = LandmarkMock(i5_x, i5_y, 0.0)
    lms[6] = LandmarkMock(i6_x, i6_y, 0.0)
    lms[7] = LandmarkMock(i7_x, i7_y, 0.0)

    # Middle finger joints (10, 11, 12)
    lms[10] = LandmarkMock(*rot(0.0, -palm_len * 1.25), 0.0)
    lms[11] = LandmarkMock(*rot(0.0, -palm_len * 1.45), 0.0)
    lms[12] = LandmarkMock(*rot(0.0, -palm_len * 1.60), 0.0)

    # Ring finger joints (13, 14, 15, 16)
    lms[13] = LandmarkMock(*rot(palm_len * 0.20, -palm_len * 0.90), 0.0)
    lms[14] = LandmarkMock(*rot(palm_len * 0.20, -palm_len * 1.20), 0.0)
    lms[15] = LandmarkMock(*rot(palm_len * 0.20, -palm_len * 1.40), 0.0)
    lms[16] = LandmarkMock(*rot(palm_len * 0.20, -palm_len * 1.55), 0.0)

    # Pinky finger joints (17, 18, 19, 20)
    lms[17] = LandmarkMock(*rot(palm_len * 0.38, -palm_len * 0.80), 0.0)
    lms[18] = LandmarkMock(*rot(palm_len * 0.40, -palm_len * 1.05), 0.0)
    lms[19] = LandmarkMock(*rot(palm_len * 0.40, -palm_len * 1.25), 0.0)
    lms[20] = LandmarkMock(*rot(palm_len * 0.40, -palm_len * 1.40), 0.0)

    return lms
