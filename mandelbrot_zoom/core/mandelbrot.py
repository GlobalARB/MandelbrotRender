"""
Core Mandelbrot iteration engine with Numba JIT compilation.

Uses smooth iteration count algorithm for continuous coloring.
"""

import numpy as np
import numba


@numba.njit(parallel=True, cache=True)
def mandelbrot_frame(
    center_real: float,
    center_imag: float,
    width: float,
    resolution_x: int,
    resolution_y: int,
    max_iter: int
) -> np.ndarray:
    """
    Compute Mandelbrot set iteration counts for a frame.

    Uses smooth iteration count algorithm for continuous coloring:
    smooth_iter = iter + 1 - log(log(|z|)) / log(2)

    Args:
        center_real: Real component of frame center
        center_imag: Imaginary component of frame center
        width: Width of view in complex plane
        resolution_x: Horizontal pixel count
        resolution_y: Vertical pixel count
        max_iter: Maximum iterations before assuming point is in set

    Returns:
        2D numpy array of smooth iteration counts (float64)
    """
    # Calculate view height maintaining aspect ratio
    aspect = resolution_x / resolution_y
    height = width / aspect

    # Output array
    result = np.zeros((resolution_y, resolution_x), dtype=np.float64)

    # Large escape radius for better smooth coloring
    escape_radius_sq = 256.0 * 256.0
    log2 = np.log(2.0)

    # Iterate over pixels in parallel
    for py in numba.prange(resolution_y):
        # Map pixel y to imaginary component
        c_imag = center_imag + (py - resolution_y / 2.0) * height / resolution_y

        for px in range(resolution_x):
            # Map pixel x to real component
            c_real = center_real + (px - resolution_x / 2.0) * width / resolution_x

            # Initialize z = 0
            z_real = 0.0
            z_imag = 0.0

            iteration = 0

            # Mandelbrot iteration: z = z^2 + c
            while iteration < max_iter:
                z_real_sq = z_real * z_real
                z_imag_sq = z_imag * z_imag

                # Check escape condition
                if z_real_sq + z_imag_sq > escape_radius_sq:
                    break

                # z = z^2 + c
                z_imag = 2.0 * z_real * z_imag + c_imag
                z_real = z_real_sq - z_imag_sq + c_real

                iteration += 1

            # Calculate smooth iteration count
            if iteration < max_iter:
                # Point escaped - calculate smooth iteration
                z_mag_sq = z_real * z_real + z_imag * z_imag
                log_zn = np.log(z_mag_sq) / 2.0  # log(|z|)
                nu = np.log(log_zn / log2) / log2
                result[py, px] = iteration + 1.0 - nu
            else:
                # Point is in set (or very slow to escape)
                result[py, px] = max_iter

    return result


@numba.njit(parallel=True, cache=True)
def mandelbrot_supersampled(
    center_real: float,
    center_imag: float,
    width: float,
    resolution_x: int,
    resolution_y: int,
    max_iter: int,
    supersample: int = 2
) -> np.ndarray:
    """
    Compute Mandelbrot with supersampling for anti-aliasing.

    Renders at supersample*resolution and averages down.

    Args:
        center_real: Real component of frame center
        center_imag: Imaginary component of frame center
        width: Width of view in complex plane
        resolution_x: Output horizontal pixel count
        resolution_y: Output vertical pixel count
        max_iter: Maximum iterations
        supersample: Supersampling factor (2 = 2x2 samples per pixel)

    Returns:
        2D numpy array of smooth iteration counts at output resolution
    """
    # Render at higher resolution
    ss_x = resolution_x * supersample
    ss_y = resolution_y * supersample

    # Calculate view dimensions
    aspect = resolution_x / resolution_y
    height = width / aspect

    # Output array at supersampled resolution
    ss_result = np.zeros((ss_y, ss_x), dtype=np.float64)

    escape_radius_sq = 256.0 * 256.0
    log2 = np.log(2.0)

    for py in numba.prange(ss_y):
        c_imag = center_imag + (py - ss_y / 2.0) * height / ss_y

        for px in range(ss_x):
            c_real = center_real + (px - ss_x / 2.0) * width / ss_x

            z_real = 0.0
            z_imag = 0.0
            iteration = 0

            while iteration < max_iter:
                z_real_sq = z_real * z_real
                z_imag_sq = z_imag * z_imag

                if z_real_sq + z_imag_sq > escape_radius_sq:
                    break

                z_imag = 2.0 * z_real * z_imag + c_imag
                z_real = z_real_sq - z_imag_sq + c_real
                iteration += 1

            if iteration < max_iter:
                z_mag_sq = z_real * z_real + z_imag * z_imag
                log_zn = np.log(z_mag_sq) / 2.0
                nu = np.log(log_zn / log2) / log2
                ss_result[py, px] = iteration + 1.0 - nu
            else:
                ss_result[py, px] = max_iter

    # Downsample by averaging blocks
    result = np.zeros((resolution_y, resolution_x), dtype=np.float64)
    ss_sq = supersample * supersample

    for y in range(resolution_y):
        for x in range(resolution_x):
            total = 0.0
            for sy in range(supersample):
                for sx in range(supersample):
                    total += ss_result[y * supersample + sy, x * supersample + sx]
            result[y, x] = total / ss_sq

    return result
