"""Timestep slider + Play/Pause/Step controls.

Emits `timestep_requested(int)` when the index should change. The main
window owns the dataset cache and calls `apply_state` on receipt.
"""

from __future__ import annotations

from typing import Optional

from PySide6.QtCore import QTimer, Qt, Signal
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

import numpy as np

from core.data_loader import TimeSeries


class TimestepBar(QWidget):
    timestep_requested = Signal(int)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._time_series: Optional[TimeSeries] = None
        self._playing = False
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._on_tick)

        layout = QVBoxLayout(self)
        controls = QHBoxLayout()
        layout.addLayout(controls)

        self.play_btn = QPushButton("Play")
        self.play_btn.setEnabled(False)
        self.play_btn.clicked.connect(self.toggle_play)
        controls.addWidget(self.play_btn)

        self.prev_btn = QPushButton("<")
        self.prev_btn.setEnabled(False)
        self.prev_btn.clicked.connect(lambda: self.step(-1))
        controls.addWidget(self.prev_btn)

        self.next_btn = QPushButton(">")
        self.next_btn.setEnabled(False)
        self.next_btn.clicked.connect(lambda: self.step(1))
        controls.addWidget(self.next_btn)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setEnabled(False)
        self.slider.valueChanged.connect(self._on_slider_value)
        controls.addWidget(self.slider, 1)

        self.time_label = QLabel("t = -")
        controls.addWidget(self.time_label)

        controls.addWidget(QLabel("FPS:"))
        self.fps_spin = QSpinBox()
        self.fps_spin.setRange(1, 120)
        self.fps_spin.setValue(15)
        self.fps_spin.valueChanged.connect(self._on_fps_changed)
        controls.addWidget(self.fps_spin)

    def set_time_series(self, ts: Optional[TimeSeries]) -> None:
        self._time_series = ts
        self.stop()

        has_frames = ts is not None and len(ts) > 0
        if has_frames:
            self.slider.blockSignals(True)
            self.slider.setRange(0, len(ts) - 1)
            self.slider.setValue(0)
            self.slider.blockSignals(False)
            self._update_time_label(0)

        for w in (self.play_btn, self.prev_btn, self.next_btn, self.slider):
            w.setEnabled(bool(has_frames and len(ts) > 1))

    def set_index(self, index: int) -> None:
        if self._time_series is None or len(self._time_series) == 0:
            return
        index = max(0, min(int(index), len(self._time_series) - 1))
        self.slider.blockSignals(True)
        self.slider.setValue(index)
        self.slider.blockSignals(False)
        self._update_time_label(index)

    def step(self, delta: int) -> None:
        if self._time_series is None:
            return
        new_idx = self.slider.value() + delta
        new_idx = max(0, min(new_idx, len(self._time_series) - 1))
        self.slider.setValue(new_idx)

    def toggle_play(self) -> None:
        if self._playing:
            self.stop()
        else:
            self.start()

    def start(self) -> None:
        if self._time_series is None or len(self._time_series) <= 1:
            return
        self._playing = True
        self.play_btn.setText("Pause")
        self._timer.start(int(1000 / max(1, self.fps_spin.value())))

    def stop(self) -> None:
        self._playing = False
        self.play_btn.setText("Play")
        self._timer.stop()

    def _on_tick(self) -> None:
        if self._time_series is None:
            return
        n = len(self._time_series)
        new_idx = (self.slider.value() + 1) % n
        self.slider.setValue(new_idx)

    def _on_slider_value(self, value: int) -> None:
        self._update_time_label(value)
        self.timestep_requested.emit(int(value))

    def _on_fps_changed(self, fps: int) -> None:
        if self._playing:
            self._timer.start(int(1000 / max(1, fps)))

    def _update_time_label(self, index: int) -> None:
        if self._time_series is None:
            self.time_label.setText("t = -")
            return
        t = float(self._time_series.times[index])
        self.time_label.setText(f"t = {t:.4g}    frame {index + 1}/{len(self._time_series)}")
