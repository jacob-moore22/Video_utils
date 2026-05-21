"""Pure file loading and block-tree extraction.

All functions here take paths or pyvista objects and return plain Python
dataclasses / numpy arrays. No Qt, no globals, no I/O side effects beyond
reading the files the caller asked for.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np
import pyvista as pv


SINGLE_DATASET_EXTS = frozenset({".vtu", ".vtk", ".vts", ".vtr", ".vtp", ".vti"})
MULTIBLOCK_EXTS = frozenset({".vtm", ".vtmb"})
TIMESERIES_EXTS = frozenset({".pvd"})

SUPPORTED_EXTS = SINGLE_DATASET_EXTS | MULTIBLOCK_EXTS | TIMESERIES_EXTS


@dataclass(frozen=True)
class TimeSeries:
    """Time-series description parsed from a .pvd collection file.

    `files` are absolute paths resolved against the pvd's directory.
    `times` is a float64 array of timestep values, same length as `files`.
    A single non-time-series file is represented as a TimeSeries of length 1
    with `times = [0.0]`.
    """

    files: tuple[Path, ...]
    times: np.ndarray
    source_path: Path

    def __len__(self) -> int:
        return len(self.files)


@dataclass
class BlockNode:
    """One node in the multiblock tree.

    `index_path` is the tuple of child indices from the root MultiBlock down
    to this node. The root has `index_path = ()`. Leaf nodes (those that hold
    an actual DataSet) have a non-None `dataset_ref`. Interior nodes have
    `dataset_ref = None` and a non-empty `children` list.

    `index_path` is the stable identity used by the actor registry: an actor
    representing a leaf block can be looked up by its index_path regardless of
    timestep, so we can update geometry in place rather than rebuild actors.
    """

    name: str
    index_path: tuple[int, ...]
    children: list["BlockNode"] = field(default_factory=list)
    dataset_ref: pv.DataSet | None = None

    @property
    def is_leaf(self) -> bool:
        return self.dataset_ref is not None

    def walk(self) -> Iterator["BlockNode"]:
        """Depth-first traversal including self."""
        yield self
        for child in self.children:
            yield from child.walk()

    def leaves(self) -> Iterator["BlockNode"]:
        for node in self.walk():
            if node.is_leaf:
                yield node


@dataclass(frozen=True)
class ArrayInfo:
    """Description of a scalar/vector array available on a dataset."""

    name: str
    association: str
    num_components: int
    data_range: tuple[float, float]

    @property
    def is_vector(self) -> bool:
        return self.num_components > 1


def load_pvd(path: Path) -> TimeSeries:
    """Parse a ParaView .pvd collection into a TimeSeries.

    The .pvd format is a small XML document of <DataSet timestep=... file=.../>
    entries. We resolve `file` attributes relative to the pvd's directory.
    """
    path = Path(path)
    tree = ET.parse(path)
    root = tree.getroot()

    datasets = root.findall(".//DataSet")
    if not datasets:
        raise ValueError(f"No <DataSet> entries found in {path}")

    base_dir = path.parent
    files: list[Path] = []
    times: list[float] = []
    for ds in datasets:
        file_attr = ds.attrib.get("file")
        if file_attr is None:
            raise ValueError(f"<DataSet> in {path} missing 'file' attribute")
        files.append((base_dir / file_attr).resolve())

        ts = ds.attrib.get("timestep", ds.attrib.get("time"))
        times.append(float(ts) if ts is not None else float(len(times)))

    return TimeSeries(
        files=tuple(files),
        times=np.asarray(times, dtype=np.float64),
        source_path=path,
    )


def load_vtm(path: Path) -> pv.MultiBlock:
    """Load a .vtm multiblock dataset."""
    result = pv.read(str(path))
    if not isinstance(result, pv.MultiBlock):
        raise TypeError(f"{path} did not load as a MultiBlock (got {type(result).__name__})")
    return result


def load_dataset(path: Path) -> pv.DataSet | pv.MultiBlock:
    """Load any supported single-frame VTK file by extension.

    PVD files are time-series collections and must be opened with `load_pvd`
    plus a per-timestep call to this function; this function rejects them.
    """
    path = Path(path)
    ext = path.suffix.lower()
    if ext in TIMESERIES_EXTS:
        raise ValueError(
            f"{path} is a time-series collection; use load_pvd and then "
            "load_dataset on each per-timestep file."
        )
    if ext not in SUPPORTED_EXTS:
        raise ValueError(f"Unsupported file extension {ext!r} for {path}")
    return pv.read(str(path))


def open_path(path: Path) -> tuple[TimeSeries, pv.DataSet | pv.MultiBlock]:
    """Top-level open: returns (time_series, first_frame_dataset).

    For a .pvd, the TimeSeries covers all timesteps and the dataset is the
    first frame. For any other supported file, the TimeSeries has a single
    entry pointing back at the same file.
    """
    path = Path(path)
    ext = path.suffix.lower()
    if ext in TIMESERIES_EXTS:
        ts = load_pvd(path)
        first = load_dataset(ts.files[0])
        return ts, first

    dataset = load_dataset(path)
    ts = TimeSeries(
        files=(path.resolve(),),
        times=np.zeros(1, dtype=np.float64),
        source_path=path,
    )
    return ts, dataset


def build_block_tree(
    data: pv.DataSet | pv.MultiBlock,
    name: str = "root",
    index_path: tuple[int, ...] = (),
) -> BlockNode:
    """Recursively turn a MultiBlock (or single DataSet) into a BlockNode tree.

    Interior nodes (MultiBlock instances) have children and no dataset.
    Leaf nodes hold a reference to the underlying pv.DataSet.
    """
    if isinstance(data, pv.MultiBlock):
        node = BlockNode(name=name, index_path=index_path)
        for i in range(data.n_blocks):
            child_name = data.get_block_name(i) or f"block_{i}"
            child_data = data[i]
            if child_data is None:
                continue
            node.children.append(
                build_block_tree(child_data, name=child_name, index_path=index_path + (i,))
            )
        return node

    return BlockNode(
        name=name,
        index_path=index_path,
        dataset_ref=data,
    )


def get_block_by_index_path(
    data: pv.DataSet | pv.MultiBlock,
    index_path: tuple[int, ...],
) -> pv.DataSet | pv.MultiBlock | None:
    """Walk a MultiBlock by index_path. Returns None if the path is invalid."""
    node: pv.DataSet | pv.MultiBlock | None = data
    for idx in index_path:
        if not isinstance(node, pv.MultiBlock):
            return None
        if idx >= node.n_blocks:
            return None
        node = node[idx]
    return node


def _iter_datasets(data: pv.DataSet | pv.MultiBlock) -> Iterator[pv.DataSet]:
    if isinstance(data, pv.MultiBlock):
        for i in range(data.n_blocks):
            child = data[i]
            if child is None:
                continue
            yield from _iter_datasets(child)
    else:
        yield data


def collect_point_cell_arrays(
    data: pv.DataSet | pv.MultiBlock,
) -> dict[str, ArrayInfo]:
    """Inventory all scalar/vector arrays available on `data`.

    For multiblock inputs, arrays are unioned across leaf datasets; the
    reported range is the min/max across all blocks that carry the array.
    Keys are unique on (association, name) with cell arrays prefixed by
    'cell:' and point arrays by 'point:' so a name collision across
    associations stays addressable.
    """
    accum: dict[str, dict] = {}

    for dataset in _iter_datasets(data):
        for association, container in (
            ("point", dataset.point_data),
            ("cell", dataset.cell_data),
        ):
            for name in container.keys():
                arr = container[name]
                if arr is None or arr.size == 0:
                    continue
                key = f"{association}:{name}"
                num_components = 1 if arr.ndim == 1 else int(arr.shape[1])
                if num_components > 1:
                    finite = arr[np.isfinite(arr).all(axis=1)]
                    if finite.size:
                        mags = np.linalg.norm(finite, axis=1)
                        lo, hi = float(mags.min()), float(mags.max())
                    else:
                        lo, hi = 0.0, 0.0
                else:
                    finite = arr[np.isfinite(arr)]
                    if finite.size:
                        lo, hi = float(finite.min()), float(finite.max())
                    else:
                        lo, hi = 0.0, 0.0

                entry = accum.setdefault(
                    key,
                    {
                        "name": name,
                        "association": association,
                        "num_components": num_components,
                        "lo": lo,
                        "hi": hi,
                    },
                )
                entry["lo"] = min(entry["lo"], lo)
                entry["hi"] = max(entry["hi"], hi)

    return {
        key: ArrayInfo(
            name=v["name"],
            association=v["association"],
            num_components=v["num_components"],
            data_range=(v["lo"], v["hi"]),
        )
        for key, v in accum.items()
    }


def block_label(node: BlockNode) -> str:
    """Human-readable label for a tree node (used by the GUI)."""
    if not node.index_path:
        return node.name or "root"
    return node.name
