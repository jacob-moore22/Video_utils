"""Video export dialog + background QThread worker.

The dialog gathers ExportSettings from the user and runs the actual encoding
on a worker thread so the UI stays responsive. It captures the *current*
SceneState (including camera) at the moment the user clicks Export.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pyvista as pv
from PySide6.QtCore import QObject, QThread, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
)

from core.data_loader import TimeSeries
from core.scene_state import SceneState
from core.video_export import ExportSettings, export_video


class _ExportWorker(QObject):
    progress = Signal(int, int)
    finished = Signal(str)
    failed = Signal(str)

    def __init__(
        self,
        state: SceneState,
        time_series: TimeSeries,
        settings: ExportSettings,
    ) -> None:
        super().__init__()
        self._state = state
        self._time_series = time_series
        self._settings = settings
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True

    def run(self) -> None:
        try:
            def _progress(current: int, total: int) -> bool:
                self.progress.emit(current, total)
                return not self._cancelled

            output = export_video(
                self._state,
                self._time_series,
                self._settings,
                progress=_progress,
            )
            self.finished.emit(str(output))
        except Exception as exc:
            self.failed.emit(str(exc))


class ExportDialog(QDialog):
    """Modal dialog that collects export settings and runs the encode."""

    def __init__(
        self,
        state: SceneState,
        time_series: TimeSeries,
        default_path: Optional[Path] = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Export Video")
        self._state = state.copy_for_export()
        self._time_series = time_series
        self._thread: Optional[QThread] = None
        self._worker: Optional[_ExportWorker] = None

        layout = QVBoxLayout(self)
        form = QFormLayout()
        layout.addLayout(form)

        default_out = default_path or Path.cwd() / "solver_render.mp4"
        self.path_edit = QLineEdit(str(default_out))
        browse = QPushButton("Browse...")
        browse.clicked.connect(self._browse_output)
        path_row = QHBoxLayout()
        path_row.addWidget(self.path_edit, 1)
        path_row.addWidget(browse)
        form.addRow("Output:", path_row)

        self.fps_spin = QSpinBox()
        self.fps_spin.setRange(1, 120)
        self.fps_spin.setValue(30)
        form.addRow("FPS:", self.fps_spin)

        res_row = QHBoxLayout()
        self.width_spin = QSpinBox()
        self.width_spin.setRange(64, 7680)
        self.width_spin.setValue(1920)
        self.width_spin.setSingleStep(2)
        self.height_spin = QSpinBox()
        self.height_spin.setRange(64, 4320)
        self.height_spin.setValue(1080)
        self.height_spin.setSingleStep(2)
        res_row.addWidget(self.width_spin)
        res_row.addWidget(QLabel("x"))
        res_row.addWidget(self.height_spin)
        form.addRow("Resolution:", res_row)

        self.lock_youtube = QCheckBox("YouTube preset (1920x1080, H.264 high, CRF 18)")
        self.lock_youtube.setChecked(True)
        self.lock_youtube.toggled.connect(self._on_lock_toggled)
        form.addRow("", self.lock_youtube)

        self.crf_spin = QSpinBox()
        self.crf_spin.setRange(0, 51)
        self.crf_spin.setValue(18)
        self.crf_spin.setEnabled(False)
        form.addRow("CRF (quality):", self.crf_spin)

        range_row = QHBoxLayout()
        self.start_spin = QSpinBox()
        self.start_spin.setRange(0, max(0, len(time_series) - 1))
        self.start_spin.setValue(0)
        self.end_spin = QSpinBox()
        self.end_spin.setRange(1, len(time_series))
        self.end_spin.setValue(len(time_series))
        range_row.addWidget(self.start_spin)
        range_row.addWidget(QLabel("to"))
        range_row.addWidget(self.end_spin)
        form.addRow("Frame range:", range_row)

        self.progress = QProgressBar()
        self.progress.setRange(0, max(1, len(time_series)))
        layout.addWidget(self.progress)

        self.status = QLabel("")
        layout.addWidget(self.status)

        self.button_box = QDialogButtonBox(QDialogButtonBox.Cancel)
        self.export_btn = self.button_box.addButton("Export", QDialogButtonBox.AcceptRole)
        self.button_box.accepted.connect(self._on_export)
        self.button_box.rejected.connect(self._on_cancel)
        layout.addWidget(self.button_box)

        self._on_lock_toggled(True)

    def _on_lock_toggled(self, checked: bool) -> None:
        for w in (self.width_spin, self.height_spin, self.crf_spin):
            w.setEnabled(not checked)
        if checked:
            self.width_spin.setValue(1920)
            self.height_spin.setValue(1080)
            self.crf_spin.setValue(18)

    def _browse_output(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Video As",
            self.path_edit.text(),
            "MP4 Video (*.mp4);;All Files (*)",
        )
        if path:
            if not path.lower().endswith(".mp4"):
                path += ".mp4"
            self.path_edit.setText(path)

    def _on_export(self) -> None:
        output = Path(self.path_edit.text())
        settings = ExportSettings(
            output_path=output,
            fps=self.fps_spin.value(),
            resolution=(self.width_spin.value(), self.height_spin.value()),
            start_frame=self.start_spin.value(),
            end_frame=self.end_spin.value(),
            crf=self.crf_spin.value(),
        )

        self.button_box.setEnabled(False)
        self.status.setText("Encoding...")

        self._thread = QThread(self)
        self._worker = _ExportWorker(self._state, self._time_series, settings)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_finished)
        self._worker.failed.connect(self._on_failed)
        self._thread.start()

    def _on_progress(self, current: int, total: int) -> None:
        self.progress.setMaximum(total)
        self.progress.setValue(current)
        self.status.setText(f"Encoded frame {current} / {total}")

    def _on_finished(self, path: str) -> None:
        self.status.setText(f"Saved: {path}")
        self._teardown_thread()
        self.button_box.setEnabled(True)
        self.accept()

    def _on_failed(self, message: str) -> None:
        self.status.setText(f"Failed: {message}")
        self._teardown_thread()
        self.button_box.setEnabled(True)

    def _on_cancel(self) -> None:
        if self._worker is not None:
            self._worker.cancel()
        self._teardown_thread()
        self.reject()

    def _teardown_thread(self) -> None:
        if self._thread is not None:
            self._thread.quit()
            self._thread.wait()
            self._thread = None
        self._worker = None
