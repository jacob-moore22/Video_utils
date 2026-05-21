"""SceneState: the single source of truth for what the view should show.

The GUI mutates a SceneState in response to user input; `apply_state` in
rendering.py reads it and reconciles the plotter to match. Keeping all view
configuration in one plain dataclass makes it trivial to serialise (for
"save view" features later) and keeps the GUI and renderer decoupled.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional


class Representation(str, Enum):
    """How block geometry should be drawn. Mirrors ParaView's vocabulary."""

    SURFACE = "Surface"
    SURFACE_WITH_EDGES = "Surface With Edges"
    WIREFRAME = "Wireframe"
    POINTS = "Points"


@dataclass
class CameraSpec:
    """Plain-data camera description.

    Matches the values pyvista's `Plotter.camera_position` and related
    properties use, so it can be applied with a few attribute writes.
    """

    position: tuple[float, float, float]
    focal_point: tuple[float, float, float]
    view_up: tuple[float, float, float]
    view_angle: float = 30.0
    parallel_projection: bool = False
    parallel_scale: float = 1.0


@dataclass
class SceneState:
    """All view configuration in one place."""

    source_path: Optional[Path] = None
    timestep_index: int = 0

    visible_blocks: set[tuple[int, ...]] = field(default_factory=set)

    active_array_key: Optional[str] = None
    active_array_component: Optional[int] = None

    colormap: str = "viridis"
    clim: Optional[tuple[float, float]] = None
    auto_clim: bool = True

    representation: Representation = Representation.SURFACE
    edge_color: tuple[float, float, float] = (0.0, 0.0, 0.0)
    point_size: float = 5.0
    line_width: float = 1.5
    background: tuple[float, float, float] = (1.0, 1.0, 1.0)

    camera: Optional[CameraSpec] = None

    def copy_for_export(self) -> "SceneState":
        """Shallow copy suitable for handing to a background export thread.

        The set of visible blocks is duplicated so a concurrent GUI edit does
        not race with the export iteration.
        """
        return SceneState(
            source_path=self.source_path,
            timestep_index=self.timestep_index,
            visible_blocks=set(self.visible_blocks),
            active_array_key=self.active_array_key,
            active_array_component=self.active_array_component,
            colormap=self.colormap,
            clim=self.clim,
            auto_clim=self.auto_clim,
            representation=self.representation,
            edge_color=self.edge_color,
            point_size=self.point_size,
            line_width=self.line_width,
            background=self.background,
            camera=self.camera,
        )
