"""
Frame rendering pipeline with anti-aliasing and dynamic iteration scaling.
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple

from ..core.mandelbrot import mandelbrot_supersampled


@dataclass
class ZoomConfig:
    """Configuration for a single zoom frame."""
    center_real: float
    center_imag: float
    width: float
    frame_number: int
    max_iter: int
    time_offset: float  # For color cycling (0.0 to 1.0)


class FrameRenderer:
    """
    Orchestrates rendering of individual frames with anti-aliasing.
    """

    def __init__(
        self,
        resolution: Tuple[int, int] = (1920, 1080),
        supersample_factor: int = 2,
        base_max_iter: int = 1000
    ):
        """
        Args:
            resolution: (width, height) tuple
            supersample_factor: Anti-aliasing factor (2 = 2x2 samples)
            base_max_iter: Starting iteration count
        """
        self.resolution = resolution
        self.supersample_factor = supersample_factor
        self.base_max_iter = base_max_iter

    def compute_max_iter_for_zoom(self, zoom_width: float) -> int:
        """
        Deeper zooms need more iterations to reveal detail.

        Scales logarithmically with zoom depth.

        Args:
            zoom_width: Current view width in complex plane

        Returns:
            Appropriate max iteration count
        """
        # Initial view is typically width=4
        zoom_factor = 4.0 / zoom_width
        extra_iter = int(100 * np.log10(zoom_factor + 1))
        return min(self.base_max_iter + extra_iter, 10000)

    def render_frame(self, config: ZoomConfig) -> np.ndarray:
        """
        Render a single frame with supersampling.

        Args:
            config: ZoomConfig with frame parameters

        Returns:
            2D numpy array of smooth iteration counts
        """
        width, height = self.resolution

        return mandelbrot_supersampled(
            center_real=config.center_real,
            center_imag=config.center_imag,
            width=config.width,
            resolution_x=width,
            resolution_y=height,
            max_iter=config.max_iter,
            supersample=self.supersample_factor
        )


def calculate_zoom_schedule(
    initial_width: float,
    final_width: float,
    num_frames: int
) -> np.ndarray:
    """
    Calculate exponential zoom schedule for smooth visual zooming.

    Uses logarithmic interpolation so zoom appears constant speed.

    Args:
        initial_width: Starting view width
        final_width: Ending view width
        num_frames: Total number of frames

    Returns:
        Array of view widths for each frame
    """
    log_initial = np.log(initial_width)
    log_final = np.log(final_width)
    log_widths = np.linspace(log_initial, log_final, num_frames)
    return np.exp(log_widths)
