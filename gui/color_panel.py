"""Color array, colormap, clim, and edge controls.

Edits `SceneState` and emits a `state_changed` signal so the main window can
call `apply_state` once per user interaction. We don't store any state of our
own; the widgets are bound to the shared SceneState.
"""

from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QCheckBox,
    QColorDialog,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from core.data_loader import ArrayInfo
from core.scene_state import Representation, SceneState


COMPONENT_LABELS = ("Magnitude", "X", "Y", "Z")
COMPONENT_VALUES = (None, 0, 1, 2)


def list_colormaps() -> list[str]:
    """Curated set of perceptually-uniform / scientific colormaps first,
    then a longer list from matplotlib so unusual choices are still available."""
    preferred = [
        "viridis", "plasma", "inferno", "magma", "cividis",
        "turbo", "coolwarm", "RdBu_r", "Spectral_r",
        "Blues", "Reds", "Greens", "Greys",
    ]

    try:
        import matplotlib
        all_maps = sorted(matplotlib.colormaps)
    except Exception:
        all_maps = []

    try:
        import cmocean  # noqa: F401
        cmocean_maps = [
            "cmo.thermal", "cmo.haline", "cmo.solar", "cmo.ice",
            "cmo.deep", "cmo.dense", "cmo.balance",
        ]
    except Exception:
        cmocean_maps = []

    seen: set[str] = set()
    ordered: list[str] = []
    for name in preferred + cmocean_maps + all_maps:
        if name in seen:
            continue
        seen.add(name)
        ordered.append(name)
    return ordered


class ColorPanel(QWidget):
    """Controls for active array and how it is colored."""

    state_changed = Signal()

    def __init__(self, state: SceneState, parent=None) -> None:
        super().__init__(parent)
        self._state = state
        self._arrays: dict[str, ArrayInfo] = {}
        self._suspend = False

        layout = QVBoxLayout(self)
        form = QFormLayout()
        layout.addLayout(form)

        self.array_combo = QComboBox()
        self.array_combo.currentIndexChanged.connect(self._on_array_changed)
        form.addRow("Array:", self.array_combo)

        self.component_combo = QComboBox()
        for label in COMPONENT_LABELS:
            self.component_combo.addItem(label)
        self.component_combo.currentIndexChanged.connect(self._on_component_changed)
        form.addRow("Component:", self.component_combo)

        self.colormap_combo = QComboBox()
        for cmap in list_colormaps():
            self.colormap_combo.addItem(cmap)
        self.colormap_combo.setCurrentText(state.colormap)
        self.colormap_combo.currentTextChanged.connect(self._on_colormap_changed)
        form.addRow("Colormap:", self.colormap_combo)

        self.auto_clim = QCheckBox("Auto range")
        self.auto_clim.setChecked(state.auto_clim)
        self.auto_clim.toggled.connect(self._on_auto_clim_toggled)
        form.addRow("", self.auto_clim)

        clim_row = QHBoxLayout()
        self.clim_min = QDoubleSpinBox()
        self.clim_max = QDoubleSpinBox()
        for sb in (self.clim_min, self.clim_max):
            sb.setRange(-1e30, 1e30)
            sb.setDecimals(6)
            sb.setSingleStep(0.1)
            sb.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.clim_min.valueChanged.connect(self._on_clim_changed)
        self.clim_max.valueChanged.connect(self._on_clim_changed)
        clim_row.addWidget(self.clim_min)
        clim_row.addWidget(QLabel("to"))
        clim_row.addWidget(self.clim_max)
        form.addRow("Range:", clim_row)

        self.representation_combo = QComboBox()
        for rep in Representation:
            self.representation_combo.addItem(rep.value)
        idx = self.representation_combo.findText(state.representation.value)
        if idx >= 0:
            self.representation_combo.setCurrentIndex(idx)
        self.representation_combo.currentIndexChanged.connect(self._on_representation_changed)
        form.addRow("Representation:", self.representation_combo)

        self.edge_color_btn = QPushButton()
        self.edge_color_btn.clicked.connect(self._pick_edge_color)
        self._refresh_edge_color_button()
        form.addRow("Edge Color:", self.edge_color_btn)

        self.line_width_spin = QDoubleSpinBox()
        self.line_width_spin.setRange(0.5, 20.0)
        self.line_width_spin.setSingleStep(0.5)
        self.line_width_spin.setDecimals(1)
        self.line_width_spin.setValue(state.line_width)
        self.line_width_spin.valueChanged.connect(self._on_line_width_changed)
        form.addRow("Line Width:", self.line_width_spin)

        self.background_btn = QPushButton()
        self.background_btn.clicked.connect(self._pick_background)
        self._refresh_background_button()
        form.addRow("Background:", self.background_btn)

        self.reset_view_btn = QPushButton("Reset Camera")
        layout.addWidget(self.reset_view_btn)

        layout.addStretch(1)
        self._update_clim_enabled()
        self._update_component_enabled()

    def set_arrays(self, arrays: dict[str, ArrayInfo]) -> None:
        """Repopulate the array selector when a new dataset is loaded."""
        self._suspend = True
        try:
            self._arrays = dict(arrays)
            self.array_combo.clear()
            self.array_combo.addItem("(none)", userData=None)
            for key, info in sorted(self._arrays.items()):
                display = f"{info.name} [{info.association}]"
                if info.is_vector:
                    display += f"  ({info.num_components}-component)"
                self.array_combo.addItem(display, userData=key)

            target_key = self._state.active_array_key
            if target_key is not None:
                idx = self.array_combo.findData(target_key)
                if idx >= 0:
                    self.array_combo.setCurrentIndex(idx)
                else:
                    self._state.active_array_key = None
                    self.array_combo.setCurrentIndex(0)
            else:
                self.array_combo.setCurrentIndex(0)

            self._sync_clim_widgets_from_state()
            self._update_component_enabled()
        finally:
            self._suspend = False

    def sync_clim_widgets(self) -> None:
        """Public hook: refresh clim spinboxes from current state.clim."""
        self._sync_clim_widgets_from_state()

    def _sync_clim_widgets_from_state(self) -> None:
        if self._state.clim is None:
            return
        lo, hi = self._state.clim
        was_blocking_min = self.clim_min.blockSignals(True)
        was_blocking_max = self.clim_max.blockSignals(True)
        try:
            self.clim_min.setValue(float(lo))
            self.clim_max.setValue(float(hi))
        finally:
            self.clim_min.blockSignals(was_blocking_min)
            self.clim_max.blockSignals(was_blocking_max)

    def _on_array_changed(self, index: int) -> None:
        if self._suspend:
            return
        key = self.array_combo.itemData(index)
        self._state.active_array_key = key
        if key is not None and key in self._arrays:
            info = self._arrays[key]
            if not info.is_vector:
                self._state.active_array_component = None
            elif self._state.active_array_component is None:
                self._state.active_array_component = None
            if self._state.auto_clim:
                self._state.clim = info.data_range
                self._sync_clim_widgets_from_state()
        self._update_component_enabled()
        self.state_changed.emit()

    def _on_component_changed(self, index: int) -> None:
        if self._suspend:
            return
        self._state.active_array_component = COMPONENT_VALUES[index]
        self.state_changed.emit()

    def _on_colormap_changed(self, name: str) -> None:
        if self._suspend:
            return
        self._state.colormap = name
        self.state_changed.emit()

    def _on_auto_clim_toggled(self, checked: bool) -> None:
        if self._suspend:
            return
        self._state.auto_clim = checked
        self._update_clim_enabled()
        self.state_changed.emit()

    def _on_clim_changed(self, _value: float) -> None:
        if self._suspend:
            return
        self._state.clim = (self.clim_min.value(), self.clim_max.value())
        self.state_changed.emit()

    def _on_representation_changed(self, index: int) -> None:
        if self._suspend:
            return
        # PySide6 round-trips `userData` through QVariant, which converts
        # str-derived enums back into plain strings. Look up by the human
        # label text instead so the enum is reconstructed reliably.
        label = self.representation_combo.itemText(index)
        try:
            rep = Representation(label)
        except ValueError:
            return
        self._state.representation = rep
        self.state_changed.emit()

    def _on_line_width_changed(self, value: float) -> None:
        if self._suspend:
            return
        self._state.line_width = float(value)
        self.state_changed.emit()

    def _pick_edge_color(self) -> None:
        r, g, b = self._state.edge_color
        initial = QColor.fromRgbF(r, g, b)
        chosen = QColorDialog.getColor(initial, self, "Edge Color")
        if not chosen.isValid():
            return
        self._state.edge_color = (chosen.redF(), chosen.greenF(), chosen.blueF())
        self._refresh_edge_color_button()
        self.state_changed.emit()

    def _refresh_edge_color_button(self) -> None:
        r, g, b = self._state.edge_color
        hex_color = f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"
        text_color = "#000000" if (r * 299 + g * 587 + b * 114) / 1000 > 0.5 else "#ffffff"
        self.edge_color_btn.setText(hex_color)
        self.edge_color_btn.setStyleSheet(
            f"QPushButton {{ background-color: {hex_color}; color: {text_color}; "
            f"border: 1px solid #888; padding: 4px; }}"
        )

    def _pick_background(self) -> None:
        r, g, b = self._state.background
        initial = QColor.fromRgbF(r, g, b)
        chosen = QColorDialog.getColor(
            initial, self, "Background Color", QColorDialog.ShowAlphaChannel
        )
        if not chosen.isValid():
            return
        self._state.background = (chosen.redF(), chosen.greenF(), chosen.blueF())
        self._refresh_background_button()
        self.state_changed.emit()

    def _refresh_background_button(self) -> None:
        r, g, b = self._state.background
        hex_color = f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"
        text_color = "#000000" if (r * 299 + g * 587 + b * 114) / 1000 > 0.5 else "#ffffff"
        self.background_btn.setText(hex_color)
        self.background_btn.setStyleSheet(
            f"QPushButton {{ background-color: {hex_color}; color: {text_color}; "
            f"border: 1px solid #888; padding: 4px; }}"
        )

    def _update_clim_enabled(self) -> None:
        enabled = not self._state.auto_clim
        self.clim_min.setEnabled(enabled)
        self.clim_max.setEnabled(enabled)

    def _update_component_enabled(self) -> None:
        key = self._state.active_array_key
        is_vector = bool(key and key in self._arrays and self._arrays[key].is_vector)
        self.component_combo.setEnabled(is_vector)
        if not is_vector:
            self._suspend = True
            try:
                self.component_combo.setCurrentIndex(0)
            finally:
                self._suspend = False
