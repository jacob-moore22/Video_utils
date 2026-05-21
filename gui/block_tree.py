"""Checkable tree widget that drives `SceneState.visible_blocks`.

Mirrors the spirit of ParaView's ExtractBlock filter: every block in the
multiblock hierarchy gets a checkbox; toggling it propagates to children and
to the live render via a Qt signal.
"""

from __future__ import annotations

from typing import Iterable, Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QMenu,
    QTreeWidget,
    QTreeWidgetItem,
    QTreeWidgetItemIterator,
)

from core.data_loader import BlockNode


_INDEX_PATH_ROLE = Qt.UserRole + 1


class BlockTree(QTreeWidget):
    """Checkable view of a multiblock dataset.

    Emits `visibility_changed(set[tuple[int, ...]])` whenever the user toggles
    a checkbox, with the full new set of visible block index_paths. The caller
    writes it straight into `state.visible_blocks` and calls `apply_state`.
    """

    visibility_changed = Signal(set)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setHeaderLabel("Blocks")
        self.setSelectionMode(QTreeWidget.SingleSelection)
        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self._show_context_menu)
        self.itemChanged.connect(self._on_item_changed)
        self._suspend_signals = False

    def populate(self, root: BlockNode, default_visible: bool = True) -> None:
        """Rebuild the tree from a BlockNode. All leaves checked by default."""
        self._suspend_signals = True
        try:
            self.clear()
            for child in root.children:
                self.addTopLevelItem(self._make_item(child, default_visible))
            self.expandToDepth(1)
        finally:
            self._suspend_signals = False
        self.visibility_changed.emit(self._collect_visible())

    def _make_item(self, node: BlockNode, default_visible: bool) -> QTreeWidgetItem:
        item = QTreeWidgetItem([node.name])
        item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
        item.setCheckState(0, Qt.Checked if default_visible else Qt.Unchecked)
        item.setData(0, _INDEX_PATH_ROLE, node.index_path)
        for child in node.children:
            item.addChild(self._make_item(child, default_visible))
        return item

    def visible_index_paths(self) -> set[tuple[int, ...]]:
        return self._collect_visible()

    def _collect_visible(self) -> set[tuple[int, ...]]:
        visible: set[tuple[int, ...]] = set()
        it = QTreeWidgetItemIterator(self, QTreeWidgetItemIterator.Checked)
        while it.value():
            item = it.value()
            data = item.data(0, _INDEX_PATH_ROLE)
            if isinstance(data, tuple):
                visible.add(data)
            it += 1
        return visible

    def _on_item_changed(self, item: QTreeWidgetItem, column: int) -> None:
        if self._suspend_signals or column != 0:
            return

        self._suspend_signals = True
        try:
            state = item.checkState(0)
            self._set_subtree_check_state(item, state)
            self._propagate_check_state_up(item)
        finally:
            self._suspend_signals = False

        self.visibility_changed.emit(self._collect_visible())

    def _set_subtree_check_state(self, item: QTreeWidgetItem, state: Qt.CheckState) -> None:
        for i in range(item.childCount()):
            child = item.child(i)
            child.setCheckState(0, state)
            self._set_subtree_check_state(child, state)

    def _propagate_check_state_up(self, item: QTreeWidgetItem) -> None:
        parent = item.parent()
        if parent is None:
            return
        child_states = {parent.child(i).checkState(0) for i in range(parent.childCount())}
        if child_states == {Qt.Checked}:
            new_state = Qt.Checked
        elif child_states == {Qt.Unchecked}:
            new_state = Qt.Unchecked
        else:
            new_state = Qt.PartiallyChecked
        parent.setCheckState(0, new_state)
        self._propagate_check_state_up(parent)

    def _show_context_menu(self, pos) -> None:
        item = self.itemAt(pos)
        if item is None:
            return
        menu = QMenu(self)
        action_only = QAction("Show only this", self)
        action_only.triggered.connect(lambda: self._show_only(item))
        menu.addAction(action_only)

        action_all = QAction("Show all", self)
        action_all.triggered.connect(self._show_all)
        menu.addAction(action_all)

        action_none = QAction("Hide all", self)
        action_none.triggered.connect(self._hide_all)
        menu.addAction(action_none)

        menu.exec(self.viewport().mapToGlobal(pos))

    def _show_only(self, item: QTreeWidgetItem) -> None:
        self._suspend_signals = True
        try:
            self._set_all(Qt.Unchecked)
            item.setCheckState(0, Qt.Checked)
            self._set_subtree_check_state(item, Qt.Checked)
            self._propagate_check_state_up(item)
        finally:
            self._suspend_signals = False
        self.visibility_changed.emit(self._collect_visible())

    def _show_all(self) -> None:
        self._suspend_signals = True
        try:
            self._set_all(Qt.Checked)
        finally:
            self._suspend_signals = False
        self.visibility_changed.emit(self._collect_visible())

    def _hide_all(self) -> None:
        self._suspend_signals = True
        try:
            self._set_all(Qt.Unchecked)
        finally:
            self._suspend_signals = False
        self.visibility_changed.emit(self._collect_visible())

    def _set_all(self, state: Qt.CheckState) -> None:
        it = QTreeWidgetItemIterator(self)
        while it.value():
            it.value().setCheckState(0, state)
            it += 1
