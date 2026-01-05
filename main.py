#!/usr/bin/env python3
"""
Mandelbrot Deep Zoom Video Generator

Generates a 1080p, 60-second video zooming into Seahorse Valley
with psychedelic color cycling.

Usage:
    python main.py              # Full render (1800 frames)
    python main.py --preview    # Quick preview (10 frames)
    python main.py --help       # Show all options
"""

import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

import numpy as np
from tqdm import tqdm

from mandelbrot_zoom.core.mandelbrot import get_backend_info
from mandelbrot_zoom.rendering.frame_renderer import (
    FrameRenderer,
    ZoomConfig,
    calculate_zoom_schedule
)
from mandelbrot_zoom.color.palette import ColorMapper, ColorCycler, PALETTES
from mandelbrot_zoom.video.assembler import VideoAssembler, FrameWriter


@dataclass
class VideoConfig:
    """Complete configuration for video generation."""
    # Target location (Mini Mandelbrot)
    center_real: float = -1.7497591
    center_imag: float = 0.0000001

    # Video specs
    resolution: tuple = (1920, 1080)
    fps: int = 30
    duration_seconds: int = 60

    # Zoom parameters
    initial_width: float = 4.0
    final_width: float = 1e-5  # Slower zoom (was 1e-10)

    # Quality settings
    supersample_factor: int = 2
    base_max_iter: int = 1000

    # Color settings
    palette_name: str = 'psychedelic_rainbow'
    color_frequency: float = 0.1
    color_cycle_speed: float = 2.0

    # Output
    output_dir: str = './output'
    output_filename: str = 'mandelbrot_zoom.mp4'
    save_frames: bool = False


def main():
    parser = argparse.ArgumentParser(
        description='Generate Mandelbrot deep zoom video',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Mode options
    parser.add_argument(
        '--preview', action='store_true',
        help='Generate quick preview (10 frames)'
    )

    # Target location
    parser.add_argument(
        '--center-real', type=float, default=-1.7497591,
        help='Real component of zoom center'
    )
    parser.add_argument(
        '--center-imag', type=float, default=0.0000001,
        help='Imaginary component of zoom center'
    )

    # Video specs
    parser.add_argument(
        '--width', type=int, default=1920,
        help='Video width in pixels'
    )
    parser.add_argument(
        '--height', type=int, default=1080,
        help='Video height in pixels'
    )
    parser.add_argument(
        '--fps', type=int, default=30,
        help='Frames per second'
    )
    parser.add_argument(
        '--duration', type=int, default=60,
        help='Video duration in seconds'
    )

    # Zoom parameters
    parser.add_argument(
        '--initial-width', type=float, default=4.0,
        help='Initial view width in complex plane'
    )
    parser.add_argument(
        '--final-width', type=float, default=1e-10,
        help='Final view width (zoom depth)'
    )

    # Quality
    parser.add_argument(
        '--supersample', type=int, default=2,
        help='Anti-aliasing factor (2 = 2x2 samples per pixel)'
    )
    parser.add_argument(
        '--max-iter', type=int, default=1000,
        help='Base maximum iterations'
    )

    # Colors
    parser.add_argument(
        '--palette', type=str, default='psychedelic_rainbow',
        choices=list(PALETTES.keys()),
        help='Color palette name'
    )
    parser.add_argument(
        '--color-frequency', type=float, default=0.1,
        help='Color cycling frequency (lower = more gradual)'
    )
    parser.add_argument(
        '--cycle-speed', type=float, default=2.0,
        help='Color animation cycles over video duration'
    )

    # Output
    parser.add_argument(
        '--output-dir', type=str, default='./output',
        help='Output directory'
    )
    parser.add_argument(
        '--output-file', type=str, default='mandelbrot_zoom.mp4',
        help='Output video filename'
    )
    parser.add_argument(
        '--save-frames', action='store_true',
        help='Also save individual PNG frames'
    )

    args = parser.parse_args()

    # Build configuration
    config = VideoConfig(
        center_real=args.center_real,
        center_imag=args.center_imag,
        resolution=(args.width, args.height),
        fps=args.fps,
        duration_seconds=args.duration,
        initial_width=args.initial_width,
        final_width=args.final_width,
        supersample_factor=args.supersample,
        base_max_iter=args.max_iter,
        palette_name=args.palette,
        color_frequency=args.color_frequency,
        color_cycle_speed=args.cycle_speed,
        output_dir=args.output_dir,
        output_filename=args.output_file,
        save_frames=args.save_frames
    )

    # Calculate total frames
    total_frames = config.fps * config.duration_seconds
    preview_duration = 5  # Preview shows 5 seconds of video
    if args.preview:
        total_frames = config.fps * preview_duration  # 150 frames for 5 sec preview
        # For preview, don't zoom as deep - just show first portion of zoom
        config.final_width = config.initial_width * (config.final_width / config.initial_width) ** (preview_duration / config.duration_seconds)
        config.output_filename = 'preview.mp4'

    print(f"{'=' * 60}")
    print(f"Mandelbrot Deep Zoom Video Generator")
    print(f"{'=' * 60}")
    print(f"Backend:       {get_backend_info()}")
    print(f"Resolution:    {config.resolution[0]}x{config.resolution[1]}")
    print(f"Frames:        {total_frames} @ {config.fps}fps")
    print(f"Duration:      {total_frames / config.fps:.1f} seconds")
    print(f"Zoom:          {config.initial_width} -> {config.final_width:.2e}")
    print(f"Center:        ({config.center_real}, {config.center_imag}i)")
    print(f"Palette:       {config.palette_name}")
    print(f"Supersampling: {config.supersample_factor}x")
    print(f"{'=' * 60}")

    # Initialize components
    renderer = FrameRenderer(
        resolution=config.resolution,
        supersample_factor=config.supersample_factor,
        base_max_iter=config.base_max_iter
    )

    palette = PALETTES[config.palette_name]
    color_mapper = ColorMapper(palette, config.color_frequency)
    color_cycler = ColorCycler(
        config.color_cycle_speed,
        config.fps,
        total_frames
    )

    # Setup output
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    frame_writer: Optional[FrameWriter] = None
    if config.save_frames:
        frame_writer = FrameWriter(output_dir / 'frames')

    video_path = output_dir / 'videos' / config.output_filename
    video_assembler = VideoAssembler(
        str(video_path),
        config.resolution,
        config.fps
    )

    # Calculate zoom schedule
    zoom_widths = calculate_zoom_schedule(
        config.initial_width,
        config.final_width,
        total_frames
    )

    # Render frames
    print(f"\nRendering {total_frames} frames...")
    video_assembler.start_pipe()

    for frame_num in tqdm(range(total_frames), desc="Rendering", unit="frame"):
        # Calculate max iterations for this zoom level
        max_iter = renderer.compute_max_iter_for_zoom(zoom_widths[frame_num])

        # Create frame config
        zoom_config = ZoomConfig(
            center_real=config.center_real,
            center_imag=config.center_imag,
            width=zoom_widths[frame_num],
            frame_number=frame_num,
            max_iter=max_iter,
            time_offset=color_cycler.get_time_offset(frame_num)
        )

        # Render iteration values
        iterations = renderer.render_frame(zoom_config)

        # Apply colors
        rgb_frame = color_mapper.apply(
            iterations,
            max_iter,
            zoom_config.time_offset
        )

        # Write to video
        video_assembler.write_frame(rgb_frame)

        # Optionally save frame image
        if frame_writer:
            frame_writer.save_frame(rgb_frame, frame_num)

    video_assembler.finish()

    print(f"\n{'=' * 60}")
    print(f"Video saved to: {video_path}")
    if config.save_frames:
        print(f"Frames saved to: {output_dir / 'frames'}")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()
