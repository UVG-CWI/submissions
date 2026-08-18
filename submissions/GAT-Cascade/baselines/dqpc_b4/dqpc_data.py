"""DQPC data helpers for the PU-Dense + recoloring baseline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import glob
import numpy as np
from plyfile import PlyData, PlyElement


RGB_NAMES = ("red", "green", "blue")


CG_TO_HE_TOKENS = (
    ("/CG/15fps/", "/HE/15fps/"),
    ("/CGv2/15fps/", "/HE/15fps/"),
    ("/CGv2_15/", "/HE/15fps/"),
    ("/CG_aligned/15fps/", "/HE/15fps/"),
    ("/consumer-grade_capture_system/CG/15fps/", "/high-end_capture_system/HE/15fps/"),
    ("/consumer-grade_capture_system/CGv2/15fps/", "/high-end_capture_system/HE/15fps/"),
    ("/consumer-grade_capture_system/CG_aligned/15fps/", "/high-end_capture_system/HE/15fps/"),
)


def default_cg_glob(dqpc_root: Path, split: str) -> str:
    """Return the first known CG glob that matches the local dataset layout."""
    candidates = [
        dqpc_root / split / "*" / "CG_aligned" / "15fps" / "*.ply",
        dqpc_root / split / "*" / "consumer-grade_capture_system" / "CG_aligned" / "15fps" / "*.ply",
        dqpc_root / "*" / "consumer-grade_capture_system" / "CG_aligned" / "15fps" / "*.ply",
        dqpc_root / split / "*" / "CGv2" / "15fps" / "*.ply",
        dqpc_root / split / "*" / "CGv2_15" / "*.ply",
        dqpc_root / split / "*" / "consumer-grade_capture_system" / "CGv2" / "15fps" / "*.ply",
        dqpc_root / split / "*" / "CG" / "15fps" / "*.ply",
        dqpc_root / split / "*" / "consumer-grade_capture_system" / "CG" / "15fps" / "*.ply",
        dqpc_root / "*" / "consumer-grade_capture_system" / "CGv2" / "15fps" / "*.ply",
        dqpc_root / "*" / "consumer-grade_capture_system" / "CG" / "15fps" / "*.ply",
    ]
    for pattern in candidates:
        if glob.glob(str(pattern)):
            return str(pattern)
    return str(candidates[0])


def infer_he_path_from_cg(cg_path: Path) -> Path | None:
    cg_text = str(cg_path)
    for cg_token, he_token in CG_TO_HE_TOKENS:
        if cg_token in cg_text:
            candidate = Path(cg_text.replace(cg_token, he_token))
            if candidate.exists():
                return candidate
    return None


def matching_he_path(root: Path, rel: Path) -> Path | None:
    direct = root / rel
    inferred = infer_he_path_from_cg(direct)
    if inferred is not None:
        return inferred
    if direct.exists():
        return direct
    return None


@dataclass(frozen=True)
class DQPCFramePair:
    cg_path: Path
    he_path: Path | None
    sequence: str
    frame_id: str


def read_ply_xyz_rgb(path: Path, *, with_rgb: bool = True) -> tuple[np.ndarray, np.ndarray | None]:
    ply = PlyData.read(str(path))
    vertex = ply["vertex"].data
    names = vertex.dtype.names or ()
    for name in ("x", "y", "z"):
        if name not in names:
            raise ValueError(f"{path} is missing vertex field '{name}'")

    xyz = np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=1).astype(np.float64)
    rgb = None
    if with_rgb and all(name in names for name in RGB_NAMES):
        # PLY RGB fields are integral, so float32 preserves their values exactly
        # while halving the resident color-buffer size versus float64.
        rgb = np.stack([vertex[name] for name in RGB_NAMES], axis=1).astype(np.float32)
    return xyz, rgb


def read_ply_xyz(path: Path) -> np.ndarray:
    xyz, _ = read_ply_xyz_rgb(path, with_rgb=False)
    return xyz


def chunk_slices(length: int, chunk_size: int) -> Iterator[slice]:
    size = max(1, int(chunk_size))
    for start in range(0, length, size):
        yield slice(start, min(start + size, length))


def write_ply_xyz(path: Path, xyz: np.ndarray, text: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    vertex = np.empty(xyz.shape[0], dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])
    vertex["x"] = xyz[:, 0].astype(np.float32)
    vertex["y"] = xyz[:, 1].astype(np.float32)
    vertex["z"] = xyz[:, 2].astype(np.float32)
    PlyData([PlyElement.describe(vertex, "vertex")], text=text).write(str(path))


def write_ply_xyz_rgb(path: Path, xyz: np.ndarray, rgb: np.ndarray, text: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rgb_u8 = np.clip(np.rint(rgb), 0, 255).astype(np.uint8)
    vertex = np.empty(
        xyz.shape[0],
        dtype=[
            ("x", "f4"),
            ("y", "f4"),
            ("z", "f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
        ],
    )
    vertex["x"] = xyz[:, 0].astype(np.float32)
    vertex["y"] = xyz[:, 1].astype(np.float32)
    vertex["z"] = xyz[:, 2].astype(np.float32)
    vertex["red"] = rgb_u8[:, 0]
    vertex["green"] = rgb_u8[:, 1]
    vertex["blue"] = rgb_u8[:, 2]
    PlyData([PlyElement.describe(vertex, "vertex")], text=text).write(str(path))


def quantize_xyz(xyz: np.ndarray, voxel_size: float, origin: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    if origin is None:
        origin = xyz.min(axis=0)
    coords = np.rint((xyz - origin[None, :]) / voxel_size).astype(np.int32)
    return coords, origin.astype(np.float64)


def dequantize_xyz(coords: np.ndarray, voxel_size: float, origin: np.ndarray) -> np.ndarray:
    return coords.astype(np.float64) * voxel_size + origin[None, :]


def unique_int_coords(coords: np.ndarray) -> np.ndarray:
    if coords.shape[0] == 0:
        return coords
    return np.unique(coords.astype(np.int32), axis=0)


def infer_sequence_and_frame(path: Path) -> tuple[str, str]:
    frame_id = path.stem
    parts = path.parts
    if "consumer-grade_capture_system" in parts:
        idx = parts.index("consumer-grade_capture_system")
        sequence = parts[idx - 1] if idx > 0 else path.parent.name
    elif "high-end_capture_system" in parts:
        idx = parts.index("high-end_capture_system")
        sequence = parts[idx - 1] if idx > 0 else path.parent.name
    elif "15fps" in parts:
        idx = parts.index("15fps")
        sequence = parts[idx - 2] if idx >= 2 else path.parent.name
    elif path.parent.name in {"CGv2_15", "CG_15", "HE_15"}:
        sequence = path.parent.parent.name
    else:
        sequence = path.parent.parent.parent.name if len(path.parents) >= 3 else path.parent.name
    return sequence, frame_id


def discover_frame_pairs(
    cg_glob: str,
    he_glob: str | None = None,
    cg_token: str = "/CG/15fps/",
    he_token: str = "/HE/15fps/",
    require_gt: bool = True,
) -> list[DQPCFramePair]:
    cg_paths = sorted(Path(p) for p in glob.glob(cg_glob))

    he_by_name: dict[str, Path] = {}
    he_by_seq_name: dict[tuple[str, str], Path] = {}
    if he_glob:
        for p in glob.glob(he_glob):
            path = Path(p)
            seq, _ = infer_sequence_and_frame(path)
            he_by_seq_name[(seq, path.name)] = path
            he_by_name[path.name] = path

    pairs: list[DQPCFramePair] = []
    for cg_path in cg_paths:
        he_path: Path | None = None
        cg_text = str(cg_path)
        if cg_token in cg_text:
            candidate = Path(cg_text.replace(cg_token, he_token))
            if candidate.exists():
                he_path = candidate
        if he_path is None:
            he_path = infer_he_path_from_cg(cg_path)
        if require_gt and he_path is None:
            sequence, frame_id = infer_sequence_and_frame(cg_path)
            he_path = he_by_seq_name.get((sequence, cg_path.name)) or he_by_name.get(cg_path.name)
        else:
            sequence, frame_id = infer_sequence_and_frame(cg_path)
            if he_path is None and he_by_name:
                he_path = he_by_seq_name.get((sequence, cg_path.name)) or he_by_name.get(cg_path.name)
        if require_gt and he_path is None:
            raise FileNotFoundError(f"Cannot find HE pair for {cg_path}")
        pairs.append(DQPCFramePair(cg_path=cg_path, he_path=he_path, sequence=sequence, frame_id=frame_id))
    return pairs


def infer_coordinate_units_per_mm_from_paths(paths: Sequence[Path]) -> float:
    """Return coordinate units corresponding to one millimeter."""
    sample_paths = list(paths)[:3]
    diagonals = []
    for path in sample_paths:
        xyz = read_ply_xyz(path)
        if xyz.shape[0] == 0:
            continue
        diagonals.append(float(np.linalg.norm(xyz.max(axis=0) - xyz.min(axis=0))))
    if not diagonals:
        return 1.0
    return 1.0 if float(np.median(diagonals)) > 50.0 else 0.001


def distance_scale_from_paths(paths: Sequence[Path], distance_unit: str) -> float:
    if distance_unit == "coordinate":
        return 1.0
    if distance_unit == "mm":
        # Coordinate units are dataset-wide; one frame avoids repeated
        # multi-million-point PLY reads in every command-line stage.
        return infer_coordinate_units_per_mm_from_paths(list(paths)[:1])
    raise ValueError(f"Unsupported distance unit: {distance_unit}")


def infer_voxel_size_from_paths(paths: Sequence[Path], requested: str | float) -> float:
    """Resolve `auto` voxel size from coordinate units.

    The DQPC HE point clouds are commonly stored in millimeter-like coordinates
    with a body-scale bbox diagonal around 1e3-3e3. Older notes used meter
    coordinates, where a 1 mm voxel is 0.001. Auto keeps both layouts usable.
    """
    if isinstance(requested, (int, float)):
        return float(requested)
    text = str(requested).strip().lower()
    if text not in {"auto", ""}:
        return float(text)
    return infer_coordinate_units_per_mm_from_paths(paths)


def kdtree_partition_indices(points: np.ndarray, max_points: int) -> list[np.ndarray]:
    indices = np.arange(points.shape[0])
    parts: list[np.ndarray] = []

    def split(idx: np.ndarray) -> None:
        if idx.size <= max_points:
            parts.append(idx)
            return
        subset = points[idx]
        dim = int(np.argmax(np.var(subset, axis=0)))
        order = np.argsort(subset[:, dim], kind="mergesort")
        sorted_idx = idx[order]
        mid = sorted_idx.size // 2
        split(sorted_idx[:mid])
        split(sorted_idx[mid:])

    split(indices)
    return parts


def random_crop_pair(
    cg_coords: np.ndarray,
    he_coords: np.ndarray,
    crop_size: int,
    max_cg_points: int,
    max_he_points: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    if crop_size > 0 and cg_coords.shape[0] > 0:
        center = cg_coords[rng.integers(0, cg_coords.shape[0])]
        half = crop_size // 2
        lo = center - half
        hi = center + half
        cg_mask = np.all((cg_coords >= lo[None, :]) & (cg_coords <= hi[None, :]), axis=1)
        he_mask = np.all((he_coords >= lo[None, :]) & (he_coords <= hi[None, :]), axis=1)
        cg_crop = cg_coords[cg_mask]
        he_crop = he_coords[he_mask]
        if cg_crop.shape[0] > 16 and he_crop.shape[0] > 16:
            cg_coords, he_coords = cg_crop, he_crop

    if max_cg_points > 0 and cg_coords.shape[0] > max_cg_points:
        idx = rng.choice(cg_coords.shape[0], max_cg_points, replace=False)
        cg_coords = cg_coords[idx]
    if max_he_points > 0 and he_coords.shape[0] > max_he_points:
        idx = rng.choice(he_coords.shape[0], max_he_points, replace=False)
        he_coords = he_coords[idx]
    return unique_int_coords(cg_coords), unique_int_coords(he_coords)


class DQPCPairDataset:
    def __init__(
        self,
        pairs: Iterable[DQPCFramePair],
        voxel_size: float,
        crop_size: int = 256,
        max_cg_points: int = 70000,
        max_he_points: int = 280000,
        seed: int = 1,
    ) -> None:
        self.pairs = list(pairs)
        self.voxel_size = voxel_size
        self.crop_size = crop_size
        self.max_cg_points = max_cg_points
        self.max_he_points = max_he_points
        self.seed = int(seed)
        self._worker_id: int | None = None
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, DQPCFramePair]:
        try:
            from torch.utils.data import get_worker_info

            worker = get_worker_info()
        except ImportError:
            worker = None
        worker_id = worker.id if worker is not None else -1
        if worker_id != self._worker_id:
            self.rng = np.random.default_rng(self.seed + worker_id + 1)
            self._worker_id = worker_id
        pair = self.pairs[index]
        if pair.he_path is None:
            raise ValueError(f"{pair.cg_path} has no HE target")

        cg_xyz = read_ply_xyz(pair.cg_path)
        he_xyz = read_ply_xyz(pair.he_path)
        origin = np.minimum(cg_xyz.min(axis=0), he_xyz.min(axis=0))
        cg_coords, _ = quantize_xyz(cg_xyz, self.voxel_size, origin)
        he_coords, _ = quantize_xyz(he_xyz, self.voxel_size, origin)
        cg_coords, he_coords = random_crop_pair(
            cg_coords,
            he_coords,
            self.crop_size,
            self.max_cg_points,
            self.max_he_points,
            self.rng,
        )
        feats = np.ones((cg_coords.shape[0], 1), dtype=np.float32)
        return cg_coords.astype(np.int32), feats, he_coords.astype(np.int32), pair


def collate_sparse_batch(items):
    import torch
    import MinkowskiEngine as ME

    coords, feats, targets, pairs = zip(*items)
    coords_batch = ME.utils.batched_coordinates(coords)
    feats_batch = torch.from_numpy(np.vstack(feats)).float()
    targets_batch = ME.utils.batched_coordinates(targets)
    return coords_batch, feats_batch, targets_batch, pairs
