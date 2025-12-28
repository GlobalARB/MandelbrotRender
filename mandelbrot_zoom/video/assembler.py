"""
Video assembly using FFmpeg for high-quality output.
"""

import subprocess
from pathlib import Path
from typing import Tuple, Optional
import numpy as np
from PIL import Image


class VideoAssembler:
    """
    Assembles rendered frames into a video using FFmpeg.

    Supports both piped streaming and frame-by-frame assembly.
    """

    def __init__(
        self,
        output_path: str,
        resolution: Tuple[int, int] = (1920, 1080),
        fps: int = 30,
        codec: str = 'libx264',
        crf: int = 18,
        preset: str = 'slow'
    ):
        """
        Args:
            output_path: Path to output video file
            resolution: (width, height) tuple
            fps: Frames per second
            codec: FFmpeg codec (libx264 for H.264)
            crf: Constant Rate Factor (18=high quality, 23=default)
            preset: Encoding speed preset (ultrafast to veryslow)
        """
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.resolution = resolution
        self.fps = fps
        self.codec = codec
        self.crf = crf
        self.preset = preset
        self._pipe_process: Optional[subprocess.Popen] = None

    def start_pipe(self):
        """Start FFmpeg process for piped input."""
        cmd = [
            'ffmpeg',
            '-y',  # Overwrite output
            '-f', 'rawvideo',
            '-vcodec', 'rawvideo',
            '-s', f'{self.resolution[0]}x{self.resolution[1]}',
            '-pix_fmt', 'rgb24',
            '-r', str(self.fps),
            '-i', '-',  # Read from stdin
            '-c:v', self.codec,
            '-crf', str(self.crf),
            '-preset', self.preset,
            '-pix_fmt', 'yuv420p',  # Compatibility
            '-movflags', '+faststart',  # Web streaming optimization
            str(self.output_path)
        ]

        self._pipe_process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE
        )

    def write_frame(self, frame: np.ndarray):
        """
        Write a single RGB frame to the video pipe.

        Args:
            frame: RGB uint8 array of shape (height, width, 3)
        """
        if self._pipe_process is None:
            self.start_pipe()

        self._pipe_process.stdin.write(frame.tobytes())

    def finish(self):
        """Close the pipe and finalize video."""
        if self._pipe_process is not None:
            self._pipe_process.stdin.close()
            self._pipe_process.wait()
            self._pipe_process = None

    def assemble_from_frames(self, frame_dir: str, pattern: str = 'frame_%05d.png'):
        """
        Assemble video from saved frame images.

        Args:
            frame_dir: Directory containing frame images
            pattern: Filename pattern (printf style)
        """
        input_pattern = str(Path(frame_dir) / pattern)

        cmd = [
            'ffmpeg',
            '-y',
            '-framerate', str(self.fps),
            '-i', input_pattern,
            '-c:v', self.codec,
            '-crf', str(self.crf),
            '-preset', self.preset,
            '-pix_fmt', 'yuv420p',
            '-movflags', '+faststart',
            str(self.output_path)
        ]

        subprocess.run(cmd, check=True)


class FrameWriter:
    """Saves individual frames as images."""

    def __init__(self, output_dir: str, format: str = 'png'):
        """
        Args:
            output_dir: Directory to save frames
            format: Image format (png, jpg)
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.format = format

    def save_frame(self, frame: np.ndarray, frame_number: int):
        """
        Save a single frame as an image.

        Args:
            frame: RGB uint8 array
            frame_number: Frame index for filename
        """
        filename = self.output_dir / f'frame_{frame_number:05d}.{self.format}'
        Image.fromarray(frame).save(filename)
