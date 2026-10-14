"""
Speed-Adaptive 1€ Filter (One Euro Filter) Implementation
Reference: Casiez, Roussel, Vogel (CHI 2012) - "1 € Filter: A Simple Speed-based
Low-pass Filter for Noisy Input in Interactive Systems"
"""

import math
from typing import Optional, Tuple


class OneEuroFilter1D:
    """First-order speed-adaptive low-pass filter to eliminate jitter without lag."""

    def __init__(
        self,
        min_cutoff: float = 1.2,
        beta: float = 0.35,
        d_cutoff: float = 1.0,
    ) -> None:
        """
        Parameters:
            min_cutoff: Minimum cutoff frequency in Hz when velocity is zero.
                        Lower values suppress stationary jitter.
            beta: Speed coefficient. Higher values reduce lag during fast motion.
            d_cutoff: Fixed cutoff frequency in Hz used to smooth signal derivative (velocity).
        """
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self.x_prev: Optional[float] = None
        self.dx_prev: float = 0.0
        self.t_prev: Optional[float] = None

    @staticmethod
    def _alpha(cutoff: float, dt: float) -> float:
        """Compute exponential smoothing factor alpha."""
        if cutoff <= 0.0 or dt <= 0.0:
            return 1.0
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def reset(self) -> None:
        """Clear internal historical state."""
        self.x_prev = None
        self.dx_prev = 0.0
        self.t_prev = None

    def filter(self, x: float, t: float) -> float:
        """
        Filter an incoming discrete coordinate signal x at timestamp t.

        Parameters:
            x: Input coordinate value.
            t: Timestamp in seconds (monotonically increasing).

        Returns:
            Stabilized smoothed coordinate value.
        """
        if self.t_prev is None or self.x_prev is None:
            self.x_prev = x
            self.t_prev = t
            self.dx_prev = 0.0
            return x

        dt = t - self.t_prev
        if dt <= 1e-6:
            return self.x_prev

        # Estimate and smooth instantaneous derivative (velocity)
        dx = (x - self.x_prev) / dt
        a_d = self._alpha(self.d_cutoff, dt)
        dx_hat = a_d * dx + (1.0 - a_d) * self.dx_prev

        # Dynamically adapt cutoff frequency based on velocity
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = self._alpha(cutoff, dt)
        x_hat = a * x + (1.0 - a) * self.x_prev

        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = t
        return x_hat


class Point2DEuroFilter:
    """Paired 2D One Euro Filter for normalized (x, y) coordinates."""

    def __init__(self, min_cutoff: float = 1.2, beta: float = 0.35, d_cutoff: float = 1.0) -> None:
        self.fx = OneEuroFilter1D(min_cutoff=min_cutoff, beta=beta, d_cutoff=d_cutoff)
        self.fy = OneEuroFilter1D(min_cutoff=min_cutoff, beta=beta, d_cutoff=d_cutoff)

    def reset(self) -> None:
        """Reset both coordinate filters."""
        self.fx.reset()
        self.fy.reset()

    def filter(self, x: float, y: float, t: float) -> Tuple[float, float]:
        """Filter 2D point (x, y) at timestamp t."""
        return self.fx.filter(x, t), self.fy.filter(y, t)
