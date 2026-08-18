"""DQPC patch utilities for GQE-Net color refinement."""

from __future__ import annotations

import glob
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from torch.utils.data import Dataset

from dqpc_data import matching_he_path, read_ply_xyz_rgb


def rgb_to_yuv(rgb: np.ndarray) -> np.ndarray:
    rgb = rgb.astype(np.float32)
    yuv = np.zeros_like(rgb, dtype=np.float32)
    yuv[:, 0] = 0.2126 * rgb[:, 0] + 0.7152 * rgb[:, 1] + 0.0722 * rgb[:, 2]
    yuv[:, 1] = -0.1146 * rgb[:, 0] - 0.3854 * rgb[:, 1] + 0.5000 * rgb[:, 2] + 128
    yuv[:, 2] = 0.5000 * rgb[:, 0] - 0.4542 * rgb[:, 1] - 0.0458 * rgb[:, 2] + 128
    return yuv


def yuv_to_rgb(yuv: np.ndarray) -> np.ndarray:
    yuv = yuv.astype(np.float32).copy()
    yuv[:, 1] -= 128
    yuv[:, 2] -= 128
    rgb = np.zeros_like(yuv, dtype=np.float32)
    rgb[:, 0] = yuv[:, 0] + 1.57480 * yuv[:, 2]
    rgb[:, 1] = yuv[:, 0] - 0.18733 * yuv[:, 1] - 0.46813 * yuv[:, 2]
    rgb[:, 2] = yuv[:, 0] + 1.85563 * yuv[:, 1]
    return np.clip(np.rint(rgb), 0, 255)


@dataclass(frozen=True)
class GQEPair:
    enhanced_path: Path
    he_path: Path
    sequence: str
    frame_id: str


def relative_path(root: Path, path: Path) -> Path:
    try:
        return path.resolve().relative_to(root.resolve())
    except ValueError:
        return Path(path.name)


def find_he_for_enhanced(enhanced_root: Path, enhanced_path: Path, he_root: Path) -> Path | None:
    rel = relative_path(enhanced_root, enhanced_path)
    return matching_he_path(he_root, rel)


def discover_gqe_pairs(enhanced_glob: str, enhanced_root: Path, he_root: Path) -> list[GQEPair]:
    pairs: list[GQEPair] = []
    for path_text in sorted(glob.glob(enhanced_glob)):
        enhanced_path = Path(path_text)
        he_path = find_he_for_enhanced(enhanced_root, enhanced_path, he_root)
        if he_path is None:
            raise FileNotFoundError(f"Cannot find HE target for {enhanced_path}")
        frame_id = enhanced_path.stem
        sequence = enhanced_path.parent.parent.parent.name if len(enhanced_path.parents) >= 3 else enhanced_path.parent.name
        pairs.append(GQEPair(enhanced_path, he_path, sequence, frame_id))
    return pairs


def normalize_patch_xyz(xyz: np.ndarray, coord_scale: float) -> np.ndarray:
    center = xyz.mean(axis=0, keepdims=True)
    scale = max(float(coord_scale), 1e-12)
    return ((xyz - center) / scale).astype(np.float32)


def nearest_he_yuv(
    enhanced_xyz: np.ndarray,
    he_tree: cKDTree,
    he_yuv: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    dist, idx = he_tree.query(enhanced_xyz, k=1, workers=-1)
    return he_yuv[idx], dist


def patch_indices(tree: cKDTree, points: np.ndarray, center: np.ndarray, patch_size: int) -> np.ndarray:
    k = min(patch_size, points.shape[0])
    _, idx = tree.query(center[None, :], k=k, workers=-1)
    idx = np.asarray(idx).reshape(-1)
    if idx.shape[0] < patch_size:
        pad = np.random.choice(idx, patch_size - idx.shape[0], replace=True)
        idx = np.concatenate([idx, pad])
    return idx.astype(np.int64)


class DQPCGQEDataset(Dataset):
    def __init__(
        self,
        pairs: list[GQEPair],
        channel: int,
        patch_size: int = 2048,
        samples_per_epoch: int = 10000,
        coord_scale: float = 1.0,
        residual: bool = False,
        target_max_distance: float = 20.0,
        far_target_weight: float = 0.2,
        max_cached_frames: int = 8,
        frame_sampling: str = "v3_random",
        patches_per_frame: int = 8,
        seed: int = 1,
    ) -> None:
        self.pairs = pairs
        self.channel = channel
        self.patch_size = patch_size
        self.samples_per_epoch = samples_per_epoch
        self.coord_scale = coord_scale
        self.residual = residual
        self.target_max_distance = target_max_distance
        self.far_target_weight = far_target_weight
        self.max_cached_frames = max(1, max_cached_frames)
        if frame_sampling not in {"v3_random", "cache_local"}:
            raise ValueError(f"Unsupported frame_sampling: {frame_sampling}")
        self.frame_sampling = frame_sampling
        self.patches_per_frame = max(1, int(patches_per_frame))
        self.seed = int(seed)
        self.epoch = 0
        self._worker_id: int | None = None
        self.rng = np.random.default_rng(seed)
        self._frame_order = np.arange(len(self.pairs), dtype=np.int64)
        self._cache: dict[int, dict] = {}
        self._cache_order: list[int] = []

    def __len__(self) -> int:
        return self.samples_per_epoch

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)
        if self.frame_sampling == "cache_local":
            rng = np.random.default_rng(self.seed + self.epoch)
            self._frame_order = rng.permutation(len(self.pairs))

    def _load_pair(self, pair_index: int) -> dict:
        if pair_index in self._cache:
            self._cache_order.remove(pair_index)
            self._cache_order.append(pair_index)
            return self._cache[pair_index]
        pair = self.pairs[pair_index]
        enh_xyz, enh_rgb = read_ply_xyz_rgb(pair.enhanced_path)
        he_xyz, he_rgb = read_ply_xyz_rgb(pair.he_path)
        if enh_rgb is None:
            raise ValueError(f"{pair.enhanced_path} is missing RGB")
        if he_rgb is None:
            raise ValueError(f"{pair.he_path} is missing RGB")
        item = {
            "enh_xyz": enh_xyz,
            "enh_yuv": rgb_to_yuv(enh_rgb),
            "enh_tree": cKDTree(enh_xyz),
            "he_yuv": rgb_to_yuv(he_rgb),
            "he_tree": cKDTree(he_xyz),
        }
        self._cache[pair_index] = item
        self._cache_order.append(pair_index)
        while len(self._cache_order) > self.max_cached_frames:
            stale = self._cache_order.pop(0)
            self._cache.pop(stale, None)
        return item

    def __getitem__(self, index: int):
        try:
            from torch.utils.data import get_worker_info

            worker = get_worker_info()
        except ImportError:
            worker = None
        worker_id = worker.id if worker is not None else -1
        if worker_id != self._worker_id:
            self.rng = np.random.default_rng(self.seed + self.epoch * 100003 + worker_id + 1)
            self._worker_id = worker_id
        if self.frame_sampling == "cache_local":
            block = index // self.patches_per_frame
            pair_index = int(self._frame_order[block % len(self._frame_order)])
        else:
            pair_index = int(self.rng.integers(0, len(self.pairs)))
        cached = self._load_pair(pair_index)
        enh_xyz = cached["enh_xyz"]
        enh_yuv = cached["enh_yuv"]

        center = enh_xyz[int(self.rng.integers(0, enh_xyz.shape[0]))]
        idx = patch_indices(cached["enh_tree"], enh_xyz, center, self.patch_size)
        patch_xyz = enh_xyz[idx]
        patch_yuv = enh_yuv[idx]
        target_yuv, target_dist = nearest_he_yuv(patch_xyz, cached["he_tree"], cached["he_yuv"])
        patch_xyz_norm = normalize_patch_xyz(patch_xyz, self.coord_scale)
        data = np.concatenate([patch_xyz_norm, patch_yuv[:, self.channel : self.channel + 1]], axis=1)
        label = target_yuv[:, self.channel : self.channel + 1]
        if self.residual:
            label = label - patch_yuv[:, self.channel : self.channel + 1]
        if self.target_max_distance > 0:
            weight = np.where(target_dist[:, None] <= self.target_max_distance, 1.0, self.far_target_weight)
        else:
            weight = np.ones_like(label, dtype=np.float32)
        return data.astype(np.float32), label.astype(np.float32), weight.astype(np.float32)
