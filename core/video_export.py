"""Offscreen 1080p video export.

Two encoder paths are tried in order:
  1. `imageio_ffmpeg.write_frames` - uses the ffmpeg binary bundled with the
     `imageio-ffmpeg` wheel. This is the only path guaranteed to work inside
     a PyInstaller `--onedir` build with no system dependencies.
  2. `subprocess` against a system `ffmpeg` on PATH. Used as a fallback only,
     so the deployed app can opt in to better presets if ffmpeg is present.

Both paths emit MP4/H.264 with YouTube-recommended settings: yuv420p pixel
format, libx264 codec, CRF 18 by default, faststart so the file streams
without a full download.
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Optional

import numpy as np
import pyvista as pv

from .data_loader import TimeSeries, load_dataset
from .rendering import ActorRegistry, apply_state
from .scene_state import SceneState


# YouTube upload guide recommends MP4 / H.264 / High profile / yuv420p / AAC
# audio (none here) / 1080p (1920x1080) / progressive / constant frame rate.
YOUTUBE_FFMPEG_ARGS: tuple[str, ...] = (
    "-c:v", "libx264",
    "-pix_fmt", "yuv420p",
    "-profile:v", "high",
    "-preset", "slow",
    "-crf", "18",
    "-movflags", "+faststart",
)


@dataclass(frozen=True)
class ExportSettings:
    output_path: Path
    fps: int = 30
    resolution: tuple[int, int] = (1920, 1080)
    start_frame: int = 0
    end_frame: Optional[int] = None
    crf: int = 18
    preset: str = "slow"

    def frame_range(self, total_frames: int) -> range:
        end = total_frames if self.end_frame is None else min(self.end_frame, total_frames)
        return range(max(0, self.start_frame), end)


def _ffmpeg_args(settings: ExportSettings) -> list[str]:
    args = list(YOUTUBE_FFMPEG_ARGS)
    for i, token in enumerate(args):
        if token == "-crf":
            args[i + 1] = str(settings.crf)
        elif token == "-preset":
            args[i + 1] = str(settings.preset)
    return args


def _render_frame(
    plotter: pv.Plotter,
    state: SceneState,
    dataset_root: pv.DataSet | pv.MultiBlock,
    registry: ActorRegistry,
) -> np.ndarray:
    """Apply state, render, and return an HxWx3 uint8 RGB frame."""
    apply_state(plotter, state, dataset_root, registry, reset_camera_if_empty=False)
    img = plotter.screenshot(return_img=True, transparent_background=False)
    return np.asarray(img, dtype=np.uint8)


def _open_writer_imageio_ffmpeg(settings: ExportSettings):
    """Try to open imageio-ffmpeg's writer. Returns the generator or None."""
    try:
        import imageio_ffmpeg
    except ImportError:
        return None

    try:
        writer = imageio_ffmpeg.write_frames(
            str(settings.output_path),
            size=settings.resolution,
            fps=settings.fps,
            codec="libx264",
            pix_fmt_in="rgb24",
            pix_fmt_out="yuv420p",
            quality=None,
            macro_block_size=1,
            output_params=[
                "-profile:v", "high",
                "-preset", settings.preset,
                "-crf", str(settings.crf),
                "-movflags", "+faststart",
            ],
        )
        writer.send(None)
        return writer
    except Exception:
        return None


def _open_writer_system_ffmpeg(settings: ExportSettings):
    """Spawn system ffmpeg reading raw RGB on stdin. Returns subprocess or None."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        return None

    width, height = settings.resolution
    cmd = [
        ffmpeg,
        "-y",
        "-loglevel", "error",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-pix_fmt", "rgb24",
        "-s", f"{width}x{height}",
        "-r", str(settings.fps),
        "-i", "-",
        *(_ffmpeg_args(settings)),
        str(settings.output_path),
    ]
    try:
        return subprocess.Popen(cmd, stdin=subprocess.PIPE)
    except OSError:
        return None


@dataclass
class _ImageioWriter:
    gen: object

    def write(self, frame: np.ndarray) -> None:
        self.gen.send(frame.tobytes())

    def close(self) -> None:
        try:
            self.gen.close()
        except Exception:
            pass


@dataclass
class _SubprocessWriter:
    proc: subprocess.Popen

    def write(self, frame: np.ndarray) -> None:
        assert self.proc.stdin is not None
        self.proc.stdin.write(frame.tobytes())

    def close(self) -> None:
        if self.proc.stdin is not None:
            self.proc.stdin.close()
        self.proc.wait()


def _open_writer(settings: ExportSettings):
    gen = _open_writer_imageio_ffmpeg(settings)
    if gen is not None:
        return _ImageioWriter(gen=gen)
    proc = _open_writer_system_ffmpeg(settings)
    if proc is not None:
        return _SubprocessWriter(proc=proc)
    raise RuntimeError(
        "No video encoder available: install `imageio-ffmpeg` "
        "or put `ffmpeg` on PATH."
    )


def export_video(
    state: SceneState,
    time_series: TimeSeries,
    settings: ExportSettings,
    *,
    plotter_factory: Optional[Callable[[tuple[int, int]], pv.Plotter]] = None,
    progress: Optional[Callable[[int, int], bool | None]] = None,
) -> Path:
    """Render the time series to an MP4 and return the output path.

    `plotter_factory` builds an offscreen plotter at the requested resolution.
    By default a fresh `pv.Plotter(off_screen=True, window_size=...)` is used.

    `progress(current, total)` is called once per frame; it may return False
    to request early termination.
    """
    if plotter_factory is None:
        plotter_factory = lambda size: pv.Plotter(off_screen=True, window_size=list(size))

    settings.output_path.parent.mkdir(parents=True, exist_ok=True)

    plotter = plotter_factory(settings.resolution)
    plotter.set_background(state.background)
    registry = ActorRegistry()
    writer = _open_writer(settings)

    try:
        frames = settings.frame_range(len(time_series))
        total = len(frames)
        for i, frame_idx in enumerate(frames):
            dataset = load_dataset(time_series.files[frame_idx])
            frame_state = state.copy_for_export()
            frame_state.timestep_index = frame_idx
            img = _render_frame(plotter, frame_state, dataset, registry)

            h, w, _ = img.shape
            if (w, h) != settings.resolution:
                img = _center_crop_or_pad(img, settings.resolution)

            writer.write(img)

            if progress is not None:
                if progress(i + 1, total) is False:
                    break
    finally:
        writer.close()
        plotter.close()

    return settings.output_path


def _center_crop_or_pad(img: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Force an image to exactly `size = (width, height)` without resampling.

    Sometimes the offscreen render window comes back a pixel or two off on
    certain VTK/OpenGL backends; center-crop or letterbox to the requested
    output to keep ffmpeg happy.
    """
    target_w, target_h = size
    h, w, _ = img.shape

    if h == target_h and w == target_w:
        return img

    out = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    src_y0 = max(0, (h - target_h) // 2)
    src_x0 = max(0, (w - target_w) // 2)
    dst_y0 = max(0, (target_h - h) // 2)
    dst_x0 = max(0, (target_w - w) // 2)
    copy_h = min(h, target_h)
    copy_w = min(w, target_w)
    out[dst_y0:dst_y0 + copy_h, dst_x0:dst_x0 + copy_w] = img[
        src_y0:src_y0 + copy_h, src_x0:src_x0 + copy_w
    ]
    return out


__all__ = [
    "ExportSettings",
    "export_video",
]
