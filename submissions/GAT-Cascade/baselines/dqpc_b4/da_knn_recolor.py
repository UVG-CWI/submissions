#!/usr/bin/env python3
"""Directional/adaptive KNN recoloring with geometry, normal, and color-consistency weights."""

from __future__ import annotations

import argparse
import glob
import json
import os
import tempfile
import time
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from dqpc_data import chunk_slices, distance_scale_from_paths, read_ply_xyz, read_ply_xyz_rgb, write_ply_xyz_rgb
from gqenet_dqpc import relative_path


def estimate_normals(xyz: np.ndarray, k: int, chunk_size: int = 200000) -> np.ndarray:
    k_eff = min(max(k, 3), xyz.shape[0])
    tree = cKDTree(xyz)
    normals = np.zeros_like(xyz, dtype=np.float64)
    for chunk in chunk_slices(xyz.shape[0], chunk_size):
        _, idx = tree.query(xyz[chunk], k=k_eff, workers=-1)
        pts = xyz[np.asarray(idx)]
        pts = pts - pts.mean(axis=1, keepdims=True)
        cov = np.einsum("nki,nkj->nij", pts, pts) / max(k_eff - 1, 1)
        eigvals, eigvecs = np.linalg.eigh(cov)
        normals[chunk] = eigvecs[np.arange(pts.shape[0]), :, np.argmin(eigvals, axis=1)]
    norm = np.linalg.norm(normals, axis=1, keepdims=True)
    return normals / np.maximum(norm, 1e-12)


def estimate_query_normals(tree: cKDTree, xyz: np.ndarray, query_xyz: np.ndarray, k: int) -> np.ndarray:
    k_eff = min(max(k, 3), xyz.shape[0])
    _, idx = tree.query(query_xyz, k=k_eff, workers=-1)
    pts = xyz[np.asarray(idx)]
    pts = pts - pts.mean(axis=1, keepdims=True)
    cov = np.einsum("nki,nkj->nij", pts, pts) / max(k_eff - 1, 1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    normals = eigvecs[np.arange(query_xyz.shape[0]), :, np.argmin(eigvals, axis=1)]
    norm = np.linalg.norm(normals, axis=1, keepdims=True)
    return normals / np.maximum(norm, 1e-12)


def exact_color_scale(
    source_tree: cKDTree,
    source_rgb: np.ndarray,
    target_xyz: np.ndarray,
    k_eff: int,
    query_chunk_size: int,
) -> float:
    fd, temp_path = tempfile.mkstemp(prefix="dqpc_color_scale_", suffix=".bin")
    os.close(fd)
    values = np.memmap(
        temp_path,
        mode="w+",
        dtype=np.float64,
        shape=(target_xyz.shape[0] * k_eff,),
    )
    count = 0
    try:
        for chunk in chunk_slices(target_xyz.shape[0], query_chunk_size):
            _, idx = source_tree.query(target_xyz[chunk], k=k_eff, workers=-1)
            if k_eff == 1:
                idx = idx[:, None]
            neighbor_rgb = source_rgb[idx].astype(np.float64)
            median_rgb = np.median(neighbor_rgb, axis=1, keepdims=True)
            color_dist = np.linalg.norm(neighbor_rgb - median_rgb, axis=2)
            positive = color_dist[color_dist > 0]
            values[count : count + positive.size] = positive
            count += positive.size
        if count == 0:
            return 1.0
        active = values[:count]
        middle = count // 2
        if count % 2:
            active.partition(middle)
            return float(active[middle])
        active.partition((middle - 1, middle))
        return 0.5 * float(active[middle - 1] + active[middle])
    finally:
        del values
        Path(temp_path).unlink(missing_ok=True)


def da_knn_recolor(
    source_xyz: np.ndarray,
    source_rgb: np.ndarray,
    target_xyz: np.ndarray,
    k: int,
    sigma: float | None,
    normal_k: int,
    normal_gamma: float,
    color_gamma: float,
    copy_distance: float | None,
    normal_chunk_size: int,
    query_chunk_size: int,
    color_scale_sample_size: int,
    color_scale_mode: str,
) -> np.ndarray:
    if source_xyz.shape[0] == 0:
        raise ValueError("source point cloud is empty")
    if target_xyz.shape[0] == 0:
        return np.empty((0, 3), dtype=np.float64)
    k_eff = min(k, source_xyz.shape[0])
    source_tree = cKDTree(source_xyz)
    source_normals = estimate_normals(source_xyz, normal_k, normal_chunk_size)
    target_tree = cKDTree(target_xyz)

    if sigma is None:
        kth = np.empty(target_xyz.shape[0], dtype=np.float64)
        for chunk in chunk_slices(target_xyz.shape[0], query_chunk_size):
            dist, _ = source_tree.query(target_xyz[chunk], k=k_eff, workers=-1)
            kth[chunk] = dist if k_eff == 1 else dist[:, -1]
        positive = kth[kth > 0]
        sigma_eff = float(np.median(positive)) if positive.size else 1.0
    else:
        sigma_eff = float(sigma)

    out = np.empty((target_xyz.shape[0], 3), dtype=np.float64)
    if color_scale_mode == "exact":
        color_scale = exact_color_scale(
            source_tree,
            source_rgb,
            target_xyz,
            k_eff,
            query_chunk_size,
        )
    elif color_scale_mode == "sampled":
        sample_count = min(target_xyz.shape[0], max(1, color_scale_sample_size))
        sample_idx = np.linspace(0, target_xyz.shape[0] - 1, sample_count, dtype=np.int64)
        _, color_idx = source_tree.query(target_xyz[sample_idx], k=k_eff, workers=-1)
        if k_eff == 1:
            color_idx = color_idx[:, None]
        sample_rgb = source_rgb[color_idx].astype(np.float64)
        sample_median = np.median(sample_rgb, axis=1, keepdims=True)
        sample_color_dist = np.linalg.norm(sample_rgb - sample_median, axis=2)
        positive_color_dist = sample_color_dist[sample_color_dist > 0]
        color_scale = float(np.median(positive_color_dist)) if positive_color_dist.size else 1.0
    else:
        raise ValueError(f"Unsupported color_scale_mode: {color_scale_mode}")

    for chunk in chunk_slices(target_xyz.shape[0], query_chunk_size):
        target_chunk = target_xyz[chunk]
        dist, idx = source_tree.query(target_chunk, k=k_eff, workers=-1)
        if k_eff == 1:
            dist = dist[:, None]
            idx = idx[:, None]
        spatial_w = np.exp(-(dist * dist) / (2.0 * max(sigma_eff, 1e-12) ** 2))
        target_normals = estimate_query_normals(target_tree, target_xyz, target_chunk, normal_k)
        neighbor_normals = source_normals[idx]
        normal_dot = np.abs(np.sum(neighbor_normals * target_normals[:, None, :], axis=2))
        normal_w = np.maximum(normal_dot, 1e-6) ** normal_gamma
        neighbor_rgb = source_rgb[idx].astype(np.float64)
        median_rgb = np.median(neighbor_rgb, axis=1, keepdims=True)
        color_dist = np.linalg.norm(neighbor_rgb - median_rgb, axis=2)
        color_w = np.exp(-color_gamma * color_dist / max(color_scale, 1e-12))
        weights = spatial_w * normal_w * color_w
        weights_sum = weights.sum(axis=1, keepdims=True)
        chunk_out = (neighbor_rgb * weights[:, :, None]).sum(axis=1) / np.maximum(weights_sum, 1e-12)
        fallback = weights_sum[:, 0] <= 1e-12
        if np.any(fallback):
            chunk_out[fallback] = source_rgb[idx[fallback, 0]]
        if copy_distance is not None and copy_distance >= 0:
            copy_mask = dist[:, 0] <= copy_distance
            if np.any(copy_mask):
                chunk_out[copy_mask] = source_rgb[idx[copy_mask, 0]]
        out[chunk] = chunk_out
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="DA-KNN recoloring for enhanced geometry")
    parser.add_argument("--source-glob", required=True)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--target-root", required=True, type=Path)
    parser.add_argument("--out-root", required=True, type=Path)
    parser.add_argument("--runtime-log", default=None, type=Path)
    parser.add_argument("--k", default=8, type=int)
    parser.add_argument("--sigma", default=None, type=float)
    parser.add_argument("--normal-k", default=16, type=int)
    parser.add_argument("--normal-gamma", default=2.0, type=float)
    parser.add_argument("--color-gamma", default=0.5, type=float)
    parser.add_argument("--copy-distance", default=0.5, type=float)
    parser.add_argument("--distance-unit", choices=("mm", "coordinate"), default="mm")
    parser.add_argument("--normal-chunk-size", default=200000, type=int)
    parser.add_argument("--query-chunk-size", default=250000, type=int)
    parser.add_argument("--color-scale-sample-size", default=200000, type=int)
    parser.add_argument("--color-scale-mode", choices=("exact", "sampled"), default="exact")
    parser.add_argument("--limit", default=0, type=int)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    files = sorted(Path(p) for p in glob.glob(args.source_glob))
    if args.limit > 0:
        files = files[: args.limit]
    if not files:
        raise RuntimeError(f"No source PLY files found from {args.source_glob}")
    if args.runtime_log:
        args.runtime_log.parent.mkdir(parents=True, exist_ok=True)
    distance_scale = distance_scale_from_paths(files, args.distance_unit)

    for src_path in files:
        tic = time.time()
        rel = relative_path(args.source_root, src_path)
        target_path = args.target_root / rel
        out_path = args.out_root / rel
        if args.skip_existing and out_path.exists():
            continue
        if not target_path.exists():
            raise FileNotFoundError(f"Missing target geometry for {src_path}: {target_path}")
        src_xyz, src_rgb = read_ply_xyz_rgb(src_path)
        target_xyz = read_ply_xyz(target_path)
        if src_rgb is None:
            raise ValueError(f"{src_path} is missing RGB")
        copy_distance = args.copy_distance * distance_scale if args.copy_distance >= 0 else None
        out_rgb = da_knn_recolor(
            src_xyz,
            src_rgb,
            target_xyz,
            args.k,
            args.sigma,
            args.normal_k,
            args.normal_gamma,
            args.color_gamma,
            copy_distance,
            args.normal_chunk_size,
            args.query_chunk_size,
            args.color_scale_sample_size,
            args.color_scale_mode,
        )
        write_ply_xyz_rgb(out_path, target_xyz, out_rgb)
        record = {
            "source": str(src_path),
            "target": str(target_path),
            "output": str(out_path),
            "points": int(target_xyz.shape[0]),
            "distance_unit": args.distance_unit,
            "resolved_copy_distance": copy_distance,
            "color_scale_mode": args.color_scale_mode,
            "seconds": round(time.time() - tic, 4),
        }
        print(record)
        if args.runtime_log:
            with args.runtime_log.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")


if __name__ == "__main__":
    main()
