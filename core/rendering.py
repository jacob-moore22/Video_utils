"""Reconcile a pyvista Plotter to match a SceneState.

The core abstraction is the `ActorRegistry`: a mapping from a block's
`index_path` to the live VTK actor representing it. Reconciliation diffs the
registry against the desired set of visible blocks and updates only what
changed. On timestep change we swap in new datasets but keep the actors,
which is what makes playback smooth.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pyvista as pv

from .data_loader import (
    ArrayInfo,
    BlockNode,
    build_block_tree,
    collect_point_cell_arrays,
    get_block_by_index_path,
)
from .scene_state import CameraSpec, Representation, SceneState


_REPRESENTATION_KWARGS = {
    Representation.SURFACE: {"style": "surface"},
    Representation.SURFACE_WITH_EDGES: {"style": "surface", "show_edges": True},
    Representation.WIREFRAME: {"style": "wireframe"},
    Representation.POINTS: {"style": "points"},
}


_VTK_REPRESENTATION_CODES = {
    Representation.SURFACE: 2,
    Representation.SURFACE_WITH_EDGES: 2,
    Representation.WIREFRAME: 1,
    Representation.POINTS: 0,
}


def _apply_representation_to_property(prop, state: SceneState) -> None:
    """Apply the current Representation to an existing actor's vtkProperty."""
    prop.SetRepresentation(_VTK_REPRESENTATION_CODES[state.representation])
    prop.SetEdgeVisibility(1 if state.representation == Representation.SURFACE_WITH_EDGES else 0)
    prop.SetEdgeColor(*state.edge_color)
    prop.SetPointSize(float(state.point_size))
    prop.SetLineWidth(float(state.line_width))
    if state.representation == Representation.POINTS:
        prop.SetRenderPointsAsSpheres(True)
    else:
        prop.SetRenderPointsAsSpheres(False)


def _representation_kwargs(state: SceneState) -> dict:
    """Translate the Representation enum into pyvista add_mesh kwargs."""
    base = dict(_REPRESENTATION_KWARGS[state.representation])
    base["edge_color"] = state.edge_color
    base["point_size"] = state.point_size
    base["line_width"] = state.line_width
    if state.representation == Representation.POINTS:
        base["render_points_as_spheres"] = True
    return base


@dataclass
class ActorRecord:
    """One entry in the actor registry.

    Stores the live actor handle, the index_path it represents, and the
    name pyvista assigned (used when removing the actor from the plotter).
    """

    index_path: tuple[int, ...]
    actor_name: str
    actor: object


@dataclass
class ActorRegistry:
    """All actors currently in the scene, keyed by block index_path."""

    actors: dict[tuple[int, ...], ActorRecord] = field(default_factory=dict)

    def clear(self, plotter: pv.Plotter) -> None:
        for record in list(self.actors.values()):
            plotter.remove_actor(record.actor_name, render=False)
        self.actors.clear()

    def remove(self, plotter: pv.Plotter, index_path: tuple[int, ...]) -> None:
        record = self.actors.pop(index_path, None)
        if record is not None:
            plotter.remove_actor(record.actor_name, render=False)


def _split_array_key(array_key: str) -> tuple[str, str]:
    """'point:node_vel' -> ('point', 'node_vel')."""
    association, _, name = array_key.partition(":")
    return association, name


def _resolve_scalars_for_dataset(
    dataset: pv.DataSet,
    array_key: Optional[str],
    component: Optional[int],
) -> tuple[Optional[np.ndarray], Optional[str], Optional[str]]:
    """Return (scalar_array, scalars_kwarg_name, association) for add_mesh.

    `component` is interpreted as:
      - None -> magnitude for vectors, raw scalar for scalars
      - 0/1/2 -> the X/Y/Z component for vectors
    The returned array is plain numpy; the caller passes it to add_mesh's
    `scalars` argument along with the dataset.
    """
    if array_key is None:
        return None, None, None

    association, name = _split_array_key(array_key)
    container = dataset.point_data if association == "point" else dataset.cell_data
    if name not in container.keys():
        return None, None, None

    arr = np.asarray(container[name])
    if arr.ndim == 2 and arr.shape[1] > 1:
        if component is None:
            scalars = np.linalg.norm(arr, axis=1)
        else:
            scalars = arr[:, int(component)]
    else:
        scalars = arr

    return scalars, name, association


def _compute_clim(
    dataset_root: pv.DataSet | pv.MultiBlock,
    array_key: str,
    component: Optional[int],
    visible_blocks: set[tuple[int, ...]],
) -> Optional[tuple[float, float]]:
    """Min/max of the active array across currently visible leaf blocks."""
    lo = np.inf
    hi = -np.inf
    found_any = False

    for index_path in visible_blocks:
        block = get_block_by_index_path(dataset_root, index_path)
        if not isinstance(block, pv.DataSet):
            continue
        scalars, _, _ = _resolve_scalars_for_dataset(block, array_key, component)
        if scalars is None or scalars.size == 0:
            continue
        finite = scalars[np.isfinite(scalars)]
        if finite.size == 0:
            continue
        lo = min(lo, float(finite.min()))
        hi = max(hi, float(finite.max()))
        found_any = True

    if not found_any:
        return None
    if lo == hi:
        hi = lo + 1.0
    return (lo, hi)


def _add_block_actor(
    plotter: pv.Plotter,
    dataset: pv.DataSet,
    index_path: tuple[int, ...],
    state: SceneState,
    clim: Optional[tuple[float, float]],
    is_first_visible: bool = False,
) -> ActorRecord:
    """Create a fresh actor for one leaf block.

    Only the first visible actor owns the scalar bar; subsequent blocks add
    their mesh without a duplicate scalar bar.

    For vector arrays we precompute the requested component/magnitude and
    attach it to the dataset as a regular named scalar array. Passing a
    *name* to `add_mesh(scalars=...)` is critical: passing the raw ndarray
    triggers a different pyvista code path that silently drops `style` and
    `show_edges` from the actor's vtkProperty.
    """
    actor_name = f"block_{'_'.join(str(i) for i in index_path) or 'root'}"

    scalars_array, scalar_name, association = _resolve_scalars_for_dataset(
        dataset, state.active_array_key, state.active_array_component
    )

    kwargs: dict = dict(
        name=actor_name,
        cmap=state.colormap,
        reset_camera=False,
        render=False,
    )
    kwargs.update(_representation_kwargs(state))
    if scalars_array is not None and scalar_name is not None:
        display_array_name = _stage_scalars_on_dataset(
            dataset, scalar_name, scalars_array, association
        )
        kwargs["scalars"] = display_array_name
        if is_first_visible:
            kwargs["scalar_bar_args"] = {"title": scalar_name}
            kwargs["show_scalar_bar"] = True
        else:
            kwargs["show_scalar_bar"] = False
        if clim is not None:
            kwargs["clim"] = clim
        kwargs["preference"] = "point" if association == "point" else "cell"
    else:
        kwargs["color"] = "lightsteelblue"
        kwargs["show_scalar_bar"] = False

    actor = plotter.add_mesh(dataset, **kwargs)
    return ActorRecord(index_path=index_path, actor_name=actor_name, actor=actor)


def _stage_scalars_on_dataset(
    dataset: pv.DataSet,
    scalar_name: str,
    scalars_array: np.ndarray,
    association: str,
) -> str:
    """Attach `scalars_array` to `dataset` and return the array name to plot.

    For scalar arrays we keep the original name. For vector components or
    magnitudes we write a derived array under a `__display__<orig>` name so
    the original array stays untouched and accessible to the GUI.
    """
    container = dataset.point_data if association == "point" else dataset.cell_data
    original = container.get(scalar_name)
    if original is not None and original.ndim == 1:
        return scalar_name
    derived_name = f"__display__{scalar_name}"
    container[derived_name] = scalars_array
    return derived_name


def _update_block_actor_fast(
    record: ActorRecord,
    dataset: pv.DataSet,
    state: SceneState,
    clim: Optional[tuple[float, float]],
) -> bool:
    """Fast in-place update used during timestep playback only.

    Updates the underlying dataset and the active scalar array on the existing
    mapper. Returns True if the update succeeded (caller keeps the actor), or
    False if a rebuild is required (caller should remove and re-add).

    Anything that affects the LUT/colormap, the property style, or the actor
    identity must go through a full rebuild via `_add_block_actor` instead -
    in-place LUT/property changes are unreliable across VTK 9.x versions.
    """
    try:
        mapper = record.actor.GetMapper()
        mapper.SetInputData(dataset)

        scalars, scalar_name, association = _resolve_scalars_for_dataset(
            dataset, state.active_array_key, state.active_array_component
        )
        if scalars is not None and scalar_name is not None:
            transient_name = f"__active_{scalar_name}"
            if association == "point":
                dataset.point_data[transient_name] = scalars
                mapper.SetScalarModeToUsePointFieldData()
            else:
                dataset.cell_data[transient_name] = scalars
                mapper.SetScalarModeToUseCellFieldData()
            mapper.SelectColorArray(transient_name)
            mapper.ScalarVisibilityOn()
            if clim is not None:
                mapper.SetScalarRange(*clim)
        else:
            mapper.ScalarVisibilityOff()
        return True
    except Exception:
        return False


def _resolve_visible_leaves(
    dataset_root: pv.DataSet | pv.MultiBlock,
    visible_blocks: set[tuple[int, ...]],
) -> dict[tuple[int, ...], pv.DataSet]:
    """For each requested visible block, expand to its leaf datasets.

    A user may check a parent block in the tree; in that case every leaf
    under it should be rendered. This walks the tree once and returns a dict
    of {leaf_index_path: leaf_dataset} for everything that should be visible.
    """
    if not visible_blocks:
        return {}

    root_node = build_block_tree(dataset_root)
    leaves: dict[tuple[int, ...], pv.DataSet] = {}

    for node in root_node.walk():
        if node.index_path not in visible_blocks:
            continue
        for leaf in node.leaves():
            if leaf.dataset_ref is not None:
                leaves[leaf.index_path] = leaf.dataset_ref

    return leaves


def apply_state(
    plotter: pv.Plotter,
    state: SceneState,
    dataset_root: pv.DataSet | pv.MultiBlock,
    registry: ActorRegistry,
    reset_camera_if_empty: bool = True,
    fast_update: bool = False,
) -> None:
    """Reconcile the plotter to match `state` for the given current dataset.

    Idempotent: calling it repeatedly with the same state is a no-op.

    `fast_update=True` is an opt-in path for timestep playback: it tries to
    update existing actors' input data without rebuilding, which is much
    faster but only works if every non-dataset attribute (colormap,
    representation, edges, visible blocks, ...) is unchanged from the
    previous call. Any UI-driven state change should leave it at False so a
    correct rebuild happens; the slight extra work is negligible.
    """
    desired_leaves = _resolve_visible_leaves(dataset_root, state.visible_blocks)

    if state.auto_clim and state.active_array_key is not None:
        clim = _compute_clim(
            dataset_root,
            state.active_array_key,
            state.active_array_component,
            set(desired_leaves.keys()),
        )
    else:
        clim = state.clim

    for index_path in list(registry.actors.keys()):
        if index_path not in desired_leaves:
            registry.remove(plotter, index_path)

    was_empty = not registry.actors

    if not fast_update:
        for index_path in list(registry.actors.keys()):
            registry.remove(plotter, index_path)
        _remove_all_scalar_bars(plotter)

    first = True
    for index_path, dataset in desired_leaves.items():
        if fast_update and index_path in registry.actors:
            record = registry.actors[index_path]
            if _update_block_actor_fast(record, dataset, state, clim):
                first = False
                continue
            registry.remove(plotter, index_path)

        record = _add_block_actor(
            plotter, dataset, index_path, state, clim, is_first_visible=first
        )
        registry.actors[index_path] = record
        first = False

    plotter.set_background(state.background)

    if state.camera is not None:
        _apply_camera(plotter, state.camera)
    elif was_empty and desired_leaves and reset_camera_if_empty:
        plotter.reset_camera(render=False)

    try:
        plotter.render_window.Modified()
    except Exception:
        pass
    plotter.render()


def _remove_all_scalar_bars(plotter: pv.Plotter) -> None:
    """Remove every scalar bar currently on the plotter, if any.

    pyvista's `remove_scalar_bar()` raises `StopIteration` when called with
    no scalar bars present (its internal `next(iter(...))` is unguarded),
    so we enumerate explicitly and tolerate any per-bar removal failures.
    """
    try:
        bars = plotter.scalar_bars
    except Exception:
        return
    try:
        titles = list(bars.keys())
    except Exception:
        return
    for title in titles:
        try:
            bars.remove_scalar_bar(title)
        except Exception:
            pass


def _apply_camera(plotter: pv.Plotter, cam: CameraSpec) -> None:
    plotter.camera_position = [cam.position, cam.focal_point, cam.view_up]
    plotter.camera.view_angle = cam.view_angle
    plotter.camera.parallel_projection = cam.parallel_projection
    plotter.camera.parallel_scale = cam.parallel_scale


def snapshot_camera(plotter: pv.Plotter) -> CameraSpec:
    """Capture the plotter's current camera into a CameraSpec."""
    cam = plotter.camera
    pos, fp, up = plotter.camera_position
    return CameraSpec(
        position=tuple(pos),
        focal_point=tuple(fp),
        view_up=tuple(up),
        view_angle=float(cam.view_angle),
        parallel_projection=bool(cam.parallel_projection),
        parallel_scale=float(cam.parallel_scale),
    )


__all__ = [
    "ActorRegistry",
    "ActorRecord",
    "apply_state",
    "snapshot_camera",
]
