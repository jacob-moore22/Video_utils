# Utilities for making videos from Fierro outputs

A PyVista-based desktop viewer for time-series VTK solver data (PVD / VTM /
VTU / VTK / VTS / VTR / VTP). Built with PySide6 + pyvistaqt.

## Features

- Open `.pvd` time-series and single-frame VTK files.
- Unstructured **and** structured mesh support.
- ExtractBlock-style tree with checkboxes for each block, sub-block, and
  per-rank piece in the multiblock hierarchy.
- Color by any point or cell array (scalar or vector); component selector
  (X / Y / Z / Magnitude) for vector arrays.
- Colormap dropdown (180+ colormaps from matplotlib, optionally cmocean).
- Manual or auto color range (clim).
- Mouse interaction: LMB drag to rotate, Shift+LMB or MMB to pan, scroll
  wheel to zoom (default VTK trackball interactor).
- Timestep slider with Play / Pause / Step buttons and configurable FPS.
- Export the current view to a 1080p YouTube-ready MP4 (H.264 high profile,
  yuv420p, CRF 18, `+faststart`), with background encoding and a progress
  bar.

## Install

Requires Python 3.10+.

```bash
python -m venv .venv
source .venv/bin/activate          # Linux / macOS
pip install -r requirements.txt
```

### Linux system packages (rarely needed)

VTK + Qt on minimal Linux installs may need a few system libraries for
OpenGL and X11/XCB:

```bash
# Fedora
sudo dnf install mesa-libGL libxkbcommon xcb-util-wm xcb-util-image \
                 xcb-util-keysyms xcb-util-renderutil

# Debian / Ubuntu
sudo apt install libgl1 libxkbcommon-x11-0 libxcb-icccm4 libxcb-image0 \
                 libxcb-keysyms1 libxcb-render-util0
```

A standard desktop install almost always already has these.

### Optional extras

| Package           | Purpose                                                       |
| ----------------- | ------------------------------------------------------------- |
| `cmocean`         | Extra perceptually-uniform scientific colormaps in the picker |
| `qtawesome`       | Icon set for toolbar buttons                                  |
| `ffmpeg` (system) | Higher-quality video encode fallback (auto-detected on PATH)  |

## Run

```bash
python vtk_to_video.py                                    # open empty
python vtk_to_video.py data/vtk/Fierro.solver0.pvd        # open file on launch
```

## Layout

```
core/         # Pure data-oriented modules (no Qt imports)
  data_loader.py     # load_pvd, load_dataset, build_block_tree, ...
  scene_state.py     # SceneState + CameraSpec dataclasses
  rendering.py       # apply_state(plotter, state, dataset, registry)
  video_export.py    # ExportSettings + export_video
gui/          # PySide6 widgets
  main_window.py
  block_tree.py
  color_panel.py
  timestep_bar.py
  export_dialog.py
vtk_to_video.py      # entry point
```

Data flows in one direction:

```
user input -> mutate SceneState -> apply_state() -> plotter renders
```

The `core/` package never imports Qt, so its functions can be tested and
driven from any frontend. The GUI mutates the shared `SceneState` and then
calls `apply_state`, which reconciles a live `ActorRegistry` against the
desired set of visible blocks - actors are updated in place across timestep
changes rather than rebuilt, which is what makes playback smooth.

## Packaging notes (PyInstaller-ready by design)

The project is structured so a future single-file build needs no source
changes. The following invariants are enforced today:

1. **No dynamic module imports.** Every `core/` and `gui/` import is at the
   top of its file so PyInstaller's static analysis finds it.
2. **No `__file__`-relative resource loads.** Any future resource (icons,
   `.ui` files, etc.) should be loaded via `importlib.resources` so they
   work inside a `--onedir` bundle.
3. **`imageio-ffmpeg` is the primary video encoder.** Its wheel ships a
   bundled `ffmpeg` binary that is collected automatically by PyInstaller;
   system `ffmpeg` is a fallback only.
4. **Single entry point** (`vtk_to_video.py`) at repo root so the spec
   file is one line.

When you eventually package, the recommended invocation is:

```bash
pip install pyinstaller
pyinstaller --name VideoUtils \
            --windowed \
            --onedir \
            --collect-all pyvista \
            --collect-all vtk \
            --collect-all imageio_ffmpeg \
            vtk_to_video.py
```

PyInstaller does not cross-compile; build on each target OS (or via GitHub
Actions runners) to produce Linux / Windows / macOS artifacts.

## Out of scope (v1)

- Filters beyond ExtractBlock (slice, clip, threshold, contour).
- Streamlines and glyphs.
- Multi-view layouts.
- Remote / server rendering. (A Trame frontend could be bolted onto the
  same `core/` package later if remote use ever matters - the GUI-agnostic
  split exists for exactly this reason.)
