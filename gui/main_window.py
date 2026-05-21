"""QMainWindow wiring: plotter, docks, menus, and event flow.

Holds the live `SceneState`, the loaded `TimeSeries`, and a small LRU-ish
cache of recently-loaded per-timestep datasets so playback is smooth.
"""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Optional

import pyvista as pv
from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (
    QDockWidget,
    QFileDialog,
    QMainWindow,
    QMessageBox,
    QStatusBar,
    QToolBar,
    QVBoxLayout,
    QWidget,
)
from pyvistaqt import QtInteractor

from core.data_loader import (
    SUPPORTED_EXTS,
    TimeSeries,
    build_block_tree,
    collect_point_cell_arrays,
    load_dataset,
    open_path,
)
from core.rendering import ActorRegistry, apply_state, snapshot_camera
from core.scene_state import SceneState

from .block_tree import BlockTree
from .color_panel import ColorPanel
from .export_dialog import ExportDialog
from .timestep_bar import TimestepBar


_DATASET_CACHE_SIZE = 8


class MainWindow(QMainWindow):
    """Application main window.

    The data flow is one-directional:
      user input -> mutates SceneState -> apply_state() -> plotter renders.
    """

    def __init__(self, initial_file: Optional[Path] = None) -> None:
        super().__init__()
        self.setWindowTitle("Solver Viewer")
        self.resize(1280, 800)

        self._state = SceneState()
        self._time_series: Optional[TimeSeries] = None
        self._current_dataset: Optional[pv.DataSet | pv.MultiBlock] = None
        self._dataset_cache: "OrderedDict[int, pv.DataSet | pv.MultiBlock]" = OrderedDict()
        self._actor_registry = ActorRegistry()

        # VTK's X11 render window reparents into the Qt widget; the widget
        # must already own a native X11 window before VTK reaches it or VTK
        # raises "BadWindow (invalid Window parameter)" under XWayland.
        container = QWidget(self)
        container.setAttribute(Qt.WA_NativeWindow, True)
        container.setAttribute(Qt.WA_DontCreateNativeAncestors, True)
        _ = container.winId()
        self.setCentralWidget(container)

        self.plotter = QtInteractor(parent=container)
        self.plotter.set_background(self._state.background)
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self.plotter.interactor)

        self._build_docks()
        self._build_menu_and_toolbar()
        self.setStatusBar(QStatusBar(self))

        if initial_file is not None:
            self._open_file(Path(initial_file))

    def _build_docks(self) -> None:
        self.block_tree = BlockTree(self)
        block_dock = QDockWidget("Blocks", self)
        block_dock.setWidget(self.block_tree)
        self.addDockWidget(Qt.LeftDockWidgetArea, block_dock)

        self.color_panel = ColorPanel(self._state, self)
        color_dock = QDockWidget("Color", self)
        color_dock.setWidget(self.color_panel)
        self.addDockWidget(Qt.RightDockWidgetArea, color_dock)

        self.timestep_bar = TimestepBar(self)
        time_dock = QDockWidget("Timestep", self)
        time_dock.setWidget(self.timestep_bar)
        self.addDockWidget(Qt.BottomDockWidgetArea, time_dock)

        self.block_tree.visibility_changed.connect(self._on_blocks_changed)
        self.color_panel.state_changed.connect(self._render_current)
        self.color_panel.reset_view_btn.clicked.connect(self._reset_camera)
        self.timestep_bar.timestep_requested.connect(self._on_timestep)

    def _build_menu_and_toolbar(self) -> None:
        menu = self.menuBar()
        file_menu = menu.addMenu("&File")

        open_action = QAction("&Open...", self)
        open_action.setShortcut(QKeySequence.Open)
        open_action.triggered.connect(self._on_open)
        file_menu.addAction(open_action)

        export_action = QAction("&Export Video...", self)
        export_action.setShortcut("Ctrl+E")
        export_action.triggered.connect(self._on_export)
        file_menu.addAction(export_action)

        screenshot_action = QAction("&Screenshot...", self)
        screenshot_action.setShortcut("Ctrl+P")
        screenshot_action.triggered.connect(self._on_screenshot)
        file_menu.addAction(screenshot_action)

        file_menu.addSeparator()
        quit_action = QAction("&Quit", self)
        quit_action.setShortcut(QKeySequence.Quit)
        quit_action.triggered.connect(self.close)
        file_menu.addAction(quit_action)

        view_menu = menu.addMenu("&View")
        reset_action = QAction("&Reset Camera", self)
        reset_action.setShortcut("R")
        reset_action.triggered.connect(self._reset_camera)
        view_menu.addAction(reset_action)

        background_action = QAction("&Background Color...", self)
        background_action.triggered.connect(self.color_panel._pick_background)
        view_menu.addAction(background_action)

        for label, rgb in (
            ("Background: White", (1.0, 1.0, 1.0)),
            ("Background: Black", (0.0, 0.0, 0.0)),
            ("Background: Light Gray", (0.9, 0.9, 0.9)),
            ("Background: Dark Slate", (0.10, 0.10, 0.12)),
        ):
            action = QAction(label, self)
            action.triggered.connect(lambda checked=False, c=rgb: self._set_background(c))
            view_menu.addAction(action)

        toolbar = QToolBar("Main", self)
        self.addToolBar(toolbar)
        toolbar.addAction(open_action)
        toolbar.addAction(export_action)
        toolbar.addAction(reset_action)
        toolbar.addAction(screenshot_action)

    def _on_open(self) -> None:
        ext_str = " ".join(f"*{e}" for e in sorted(SUPPORTED_EXTS))
        path_str, _ = QFileDialog.getOpenFileName(
            self,
            "Open VTK Data",
            "",
            f"VTK Files ({ext_str});;All Files (*)",
        )
        if path_str:
            self._open_file(Path(path_str))

    def _open_file(self, path: Path) -> None:
        try:
            time_series, first_dataset = open_path(path)
        except Exception as exc:
            QMessageBox.critical(self, "Open Failed", f"Could not open {path}:\n{exc}")
            return

        self._actor_registry.clear(self.plotter)
        self._dataset_cache.clear()

        self._time_series = time_series
        self._dataset_cache[0] = first_dataset
        self._current_dataset = first_dataset

        self._state.source_path = path
        self._state.timestep_index = 0
        self._state.camera = None

        root_node = build_block_tree(first_dataset)
        self.block_tree.populate(root_node, default_visible=True)
        self._state.visible_blocks = self.block_tree.visible_index_paths()

        arrays = collect_point_cell_arrays(first_dataset)
        if self._state.active_array_key is None or self._state.active_array_key not in arrays:
            self._state.active_array_key = _pick_default_array(arrays)
            if self._state.active_array_key is not None and self._state.auto_clim:
                self._state.clim = arrays[self._state.active_array_key].data_range
        self.color_panel.set_arrays(arrays)

        self.timestep_bar.set_time_series(time_series)
        self.statusBar().showMessage(f"Opened {path}")
        self._render_current(reset_camera=True)

    def _on_blocks_changed(self, visible: set) -> None:
        self._state.visible_blocks = set(visible)
        self._render_current()

    def _on_timestep(self, index: int) -> None:
        if self._time_series is None:
            return
        try:
            dataset = self._dataset_for_index(index)
        except Exception as exc:
            self.statusBar().showMessage(f"Load failed: {exc}")
            return
        self._state.timestep_index = index
        self._current_dataset = dataset
        self._render_current(fast_update=True)

    def _dataset_for_index(self, index: int) -> pv.DataSet | pv.MultiBlock:
        if index in self._dataset_cache:
            self._dataset_cache.move_to_end(index)
            return self._dataset_cache[index]
        assert self._time_series is not None
        dataset = load_dataset(self._time_series.files[index])
        self._dataset_cache[index] = dataset
        while len(self._dataset_cache) > _DATASET_CACHE_SIZE:
            self._dataset_cache.popitem(last=False)
        return dataset

    def _render_current(self, reset_camera: bool = False, fast_update: bool = False) -> None:
        if self._current_dataset is None:
            return
        if reset_camera:
            self._state.camera = None
        else:
            self._state.camera = snapshot_camera(self.plotter)
        apply_state(
            self.plotter,
            self._state,
            self._current_dataset,
            self._actor_registry,
            reset_camera_if_empty=reset_camera,
            fast_update=fast_update,
        )
        if reset_camera:
            self.plotter.reset_camera()
        self.color_panel.sync_clim_widgets()
        try:
            self.plotter.interactor.update()
        except Exception:
            pass

    def _reset_camera(self) -> None:
        self._state.camera = None
        self.plotter.reset_camera()

    def _set_background(self, rgb: tuple[float, float, float]) -> None:
        self._state.background = rgb
        self.color_panel._refresh_background_button()
        self._render_current()

    def _on_export(self) -> None:
        if self._time_series is None or self._current_dataset is None:
            QMessageBox.information(self, "Export Video", "Open a dataset first.")
            return
        self._state.camera = snapshot_camera(self.plotter)
        default_path = Path.cwd() / (
            (self._state.source_path.stem if self._state.source_path else "render") + ".mp4"
        )
        dialog = ExportDialog(self._state, self._time_series, default_path, self)
        dialog.exec()

    def _on_screenshot(self) -> None:
        path_str, _ = QFileDialog.getSaveFileName(
            self,
            "Save Screenshot",
            str(Path.cwd() / "screenshot.png"),
            "PNG Image (*.png);;All Files (*)",
        )
        if not path_str:
            return
        if not path_str.lower().endswith(".png"):
            path_str += ".png"
        self.plotter.screenshot(path_str)
        self.statusBar().showMessage(f"Saved {path_str}")

    def closeEvent(self, event) -> None:
        self.timestep_bar.stop()
        self.plotter.close()
        super().closeEvent(event)


_PREFERRED_ARRAY_NAMES = ("den", "pres", "sie", "mass", "node_vel")


def _pick_default_array(arrays: dict) -> Optional[str]:
    """Choose a sensible default array to color by on first load."""
    for name in _PREFERRED_ARRAY_NAMES:
        for key, info in arrays.items():
            if info.name == name:
                return key
    if arrays:
        return next(iter(sorted(arrays)))
    return None
