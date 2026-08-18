#!/usr/bin/env python3
"""Gaussian KNN recoloring for the DQPC PU-Dense -> GQE-Net baseline.

Transfers RGB from a colored source point cloud to a target geometry.

Expected input PLY fields:
  source: x, y, z, red, green, blue
  target: x, y, z

The output PLY contains target x/y/z with transferred RGB.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from dqpc_data import chunk_slices, distance_scale_from_paths, read_ply_xyz, read_ply_xyz_rgb, write_ply_xyz_rgb


def gaussian_recolor(
    source_xyz: np.ndarray,
    source_rgb: np.ndarray,
    target_xyz: np.ndarray,
    k: int,
    sigma: float | None,
    copy_distance: float | None = 0.5,
    query_chunk_size: int = 250000,
    eps: float = 1e-12,
) -> np.ndarray:
    if source_xyz.shape[0] == 0:
        raise ValueError("source point cloud is empty")
    if target_xyz.shape[0] == 0:
        return np.empty((0, 3), dtype=np.float64)

    k_eff = min(k, source_xyz.shape[0])
    tree = cKDTree(source_xyz)

    if sigma is None:
        # Preserve the V3 global median-sigma rule without retaining the full
        # target_count x k neighbor matrix in memory.
        kth = np.empty(target_xyz.shape[0], dtype=np.float64)
        for chunk in chunk_slices(target_xyz.shape[0], query_chunk_size):
            dist, _ = tree.query(target_xyz[chunk], k=k_eff, workers=-1)
            kth[chunk] = dist if k_eff == 1 else dist[:, -1]
        finite = kth[np.isfinite(kth) & (kth > 0)]
        sigma_eff = float(np.median(finite)) if finite.size else 1.0
    else:
        sigma_eff = float(sigma)

    sigma_eff = max(sigma_eff, eps)
    mapped_rgb = np.empty((target_xyz.shape[0], 3), dtype=np.float64)
    for chunk in chunk_slices(target_xyz.shape[0], query_chunk_size):
        dist, idx = tree.query(target_xyz[chunk], k=k_eff, workers=-1)
        if k_eff == 1:
            dist = dist[:, None]
            idx = idx[:, None]
        weights = np.exp(-(dist * dist) / (2.0 * sigma_eff * sigma_eff))
        weights_sum = weights.sum(axis=1, keepdims=True)
        nearest_rgb = source_rgb[idx[:, 0]]
        chunk_rgb = (source_rgb[idx] * weights[:, :, None]).sum(axis=1) / np.maximum(weights_sum, eps)
        bad = weights_sum[:, 0] <= eps
        if np.any(bad):
            chunk_rgb[bad] = nearest_rgb[bad]
        if copy_distance is not None and copy_distance >= 0:
            copy_mask = dist[:, 0] <= copy_distance
            if np.any(copy_mask):
                chunk_rgb[copy_mask] = nearest_rgb[copy_mask]
        mapped_rgb[chunk] = chunk_rgb
    return mapped_rgb


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gaussian KNN recoloring for enhanced point clouds")
    parser.add_argument("--source", required=True, type=Path, help="Colored source CG PLY")
    parser.add_argument("--target", required=True, type=Path, help="Enhanced geometry PLY")
    parser.add_argument("--output", required=True, type=Path, help="Output colored PLY")
    parser.add_argument("--k", default=8, type=int, help="Number of source neighbors")
    parser.add_argument("--sigma", default=None, type=float, help="Gaussian sigma in coordinate units")
    parser.add_argument("--copy-distance", default=0.5, type=float, help="Copy nearest RGB directly within this distance; interpreted by --distance-unit")
    parser.add_argument("--distance-unit", choices=("mm", "coordinate"), default="mm")
    parser.add_argument("--query-chunk-size", default=250000, type=int)
    parser.add_argument("--ascii", action="store_true", help="Write ASCII PLY instead of binary")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_xyz, source_rgb = read_ply_xyz_rgb(args.source)
    if source_rgb is None:
        raise ValueError(f"{args.source} has no RGB fields")

    target_xyz = read_ply_xyz(args.target)
    distance_scale = distance_scale_from_paths([args.source], args.distance_unit)
    copy_distance = args.copy_distance * distance_scale if args.copy_distance >= 0 else None
    target_rgb = gaussian_recolor(
        source_xyz,
        source_rgb,
        target_xyz,
        args.k,
        args.sigma,
        copy_distance=copy_distance,
        query_chunk_size=args.query_chunk_size,
    )
    write_ply_xyz_rgb(args.output, target_xyz, target_rgb, text=args.ascii)


if __name__ == "__main__":
    main()
