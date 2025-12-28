"""
Color palette system using cosine gradients for psychedelic effects.

Based on Inigo Quilez's cosine palette technique:
color(t) = a + b * cos(2*pi*(c*t + d))
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple


@dataclass
class CosineGradient:
    """
    Procedural gradient using cosine interpolation.

    Formula: color(t) = a + b * cos(2*pi*(c*t + d))

    Where a, b, c, d are RGB vectors that control:
    - a: Brightness offset (center of oscillation)
    - b: Amplitude of color variation
    - c: Frequency of color cycling
    - d: Phase offset for each channel
    """
    a: np.ndarray  # Shape (3,)
    b: np.ndarray
    c: np.ndarray
    d: np.ndarray

    def sample(self, t: np.ndarray) -> np.ndarray:
        """
        Sample the gradient at positions t.

        Args:
            t: Array of positions (can extend beyond [0,1] for wrapping)

        Returns:
            RGB array of shape (*t.shape, 3) with values in [0, 1]
        """
        t = np.asarray(t)[..., np.newaxis]
        color = self.a + self.b * np.cos(2 * np.pi * (self.c * t + self.d))
        return np.clip(color, 0, 1)


# Pre-defined psychedelic palettes
PALETTES = {
    'psychedelic_rainbow': CosineGradient(
        a=np.array([0.5, 0.5, 0.5]),
        b=np.array([0.5, 0.5, 0.5]),
        c=np.array([1.0, 1.0, 1.0]),
        d=np.array([0.0, 0.33, 0.67])
    ),
    'electric_plasma': CosineGradient(
        a=np.array([0.5, 0.5, 0.5]),
        b=np.array([0.5, 0.5, 0.5]),
        c=np.array([2.0, 1.0, 0.5]),
        d=np.array([0.5, 0.2, 0.25])
    ),
    'deep_ocean': CosineGradient(
        a=np.array([0.2, 0.4, 0.6]),
        b=np.array([0.3, 0.4, 0.4]),
        c=np.array([1.5, 1.0, 0.7]),
        d=np.array([0.0, 0.15, 0.35])
    ),
    'fire_ice': CosineGradient(
        a=np.array([0.5, 0.3, 0.5]),
        b=np.array([0.5, 0.5, 0.5]),
        c=np.array([1.0, 1.5, 2.0]),
        d=np.array([0.0, 0.5, 0.33])
    ),
    'neon_dreams': CosineGradient(
        a=np.array([0.6, 0.3, 0.7]),
        b=np.array([0.4, 0.6, 0.3]),
        c=np.array([1.5, 0.8, 1.2]),
        d=np.array([0.2, 0.6, 0.1])
    ),
    'sunset_glow': CosineGradient(
        a=np.array([0.5, 0.3, 0.3]),
        b=np.array([0.5, 0.4, 0.3]),
        c=np.array([1.0, 0.7, 0.4]),
        d=np.array([0.0, 0.15, 0.2])
    ),
}


class ColorMapper:
    """Maps iteration values to RGB colors using a palette with cycling."""

    def __init__(
        self,
        palette: CosineGradient,
        color_frequency: float = 0.1,
        interior_color: Tuple[int, int, int] = (0, 0, 0)
    ):
        """
        Args:
            palette: CosineGradient to use for coloring
            color_frequency: How quickly colors cycle (lower = more gradual)
            interior_color: RGB tuple for interior of Mandelbrot set
        """
        self.palette = palette
        self.frequency = color_frequency
        self.interior = np.array(interior_color) / 255.0

    def apply(
        self,
        iterations: np.ndarray,
        max_iter: int,
        time_offset: float = 0.0
    ) -> np.ndarray:
        """
        Convert iteration counts to RGB image.

        Args:
            iterations: 2D array of smooth iteration counts
            max_iter: Maximum iteration value
            time_offset: Phase offset for color cycling animation (0.0 to 1.0)

        Returns:
            RGB image as uint8 array of shape (height, width, 3)
        """
        # Normalize iterations with log scaling for smoother gradients
        # and add time offset for cycling animation
        t = np.log(iterations + 1) * self.frequency + time_offset

        # Sample palette
        colors = self.palette.sample(t)

        # Set interior points to interior color
        interior_mask = iterations >= max_iter - 0.5
        colors[interior_mask] = self.interior

        # Convert to uint8
        return (colors * 255).astype(np.uint8)


class ColorCycler:
    """Manages color cycling animation over time."""

    def __init__(
        self,
        cycle_speed: float = 1.0,
        fps: int = 30,
        total_frames: int = 1800
    ):
        """
        Args:
            cycle_speed: Number of complete color cycles over the video
            fps: Frames per second
            total_frames: Total frames in video
        """
        self.cycle_speed = cycle_speed
        self.fps = fps
        self.total_frames = total_frames

    def get_time_offset(self, frame_number: int) -> float:
        """
        Calculate color phase offset for a given frame.

        Returns value in [0, 1) that wraps around.
        """
        progress = frame_number / self.total_frames
        return (progress * self.cycle_speed) % 1.0
