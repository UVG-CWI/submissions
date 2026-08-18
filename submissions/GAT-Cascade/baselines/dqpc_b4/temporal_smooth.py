#!/usr/bin/env python3
"""Lightweight temporal color smoothing for DQPC colored PLY sequences."""

from __future__ import annotations

import argparse
import glob
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from dqpc_data import chunk_slices, distance_scale_from_paths, read_ply_xyz_rgb, write_ply_xyz_rgb
from gqenet_dqpc import relative_path


def sequence_key(input_root: Path, path: Path) -> Path:
    rel = relative_path(input_root, path)
    return rel.parent


def map_colors(
    ref_xyz: np.ndarray,
    ref_rgb: np.ndarray,
    cur_xyz: np.ndarray,
    cur_rgb: np.ndarray,
    max_distance: float | None,
    max_color_delta: float | None,
    query_chunk_size: int,
    ref_tree: cKDTree | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    tree = ref_tree or cKDTree(ref_xyz)
    mapped = np.empty_like(cur_rgb)
    valid = np.empty(cur_xyz.shape[0], dtype=bool)
    for chunk in chunk_slices(cur_xyz.shape[0], query_chunk_size):
        dist, idx = tree.query(cur_xyz[chunk], k=1, workers=-1)
        mapped_chunk = ref_rgb[idx]
        valid_chunk = np.ones(idx.shape[0], dtype=bool)
        if max_distance is not None and max_distance > 0:
            valid_chunk &= dist <= max_distance
        if max_color_delta is not None and max_color_delta > 0:
            color_dist = np.linalg.norm(
                mapped_chunk.astype(np.float64) - cur_rgb[chunk].astype(np.float64),
                axis=1,
            )
            valid_chunk &= color_dist <= max_color_delta
        mapped[chunk] = mapped_chunk
        valid[chunk] = valid_chunk
    return mapped, valid


def smooth_frame(
    prev_data,
    cur_data,
    next_data,
    alpha: float,
    max_distance: float | None,
    max_color_delta: float | None,
    query_chunk_size: int,
) -> np.ndarray:
    cur_xyz, cur_rgb, _ = cur_data
    accum = cur_rgb.astype(np.float64) * alpha
    weights = np.full((cur_rgb.shape[0], 1), alpha, dtype=np.float64)
    side_weight = (1.0 - alpha) / float((prev_data is not None) + (next_data is not None) or 1)

    for data in (prev_data, next_data):
        if data is None:
            continue
        ref_xyz, ref_rgb, ref_tree = data
        mapped, valid = map_colors(
            ref_xyz,
            ref_rgb,
            cur_xyz,
            cur_rgb,
            max_distance,
            max_color_delta,
            query_chunk_size,
            ref_tree,
        )
        accum[valid] += mapped[valid] * side_weight
        weights[valid] += side_weight
    return accum / np.maximum(weights, 1e-12)


def main() -> None:
    parser = argparse.ArgumentParser(description="Temporal color smoothing for colored PLY sequences")
    parser.add_argument("--input-root", required=True, type=Path)
    parser.add_argument("--input-glob", required=True)
    parser.add_argument("--out-root", required=True, type=Path)
    parser.add_argument("--runtime-log", default=None, type=Path)
    parser.add_argument("--alpha", default=0.75, type=float, help="Current-frame RGB weight")
    parser.add_argument("--max-distance", default=30.0, type=float, help="Reject temporal matches farther than this distance; interpreted by --distance-unit")
    parser.add_argument("--distance-unit", choices=("mm", "coordinate"), default="mm")
    parser.add_argument("--max-color-delta", default=35.0, type=float, help="Reject temporal matches with RGB L2 difference above this value; <=0 disables")
    parser.add_argument("--query-chunk-size", default=250000, type=int)
    parser.add_argument("--limit", default=0, type=int)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    files = sorted(Path(p) for p in glob.glob(args.input_glob))
    if args.limit > 0:
        files = files[: args.limit]
    if not files:
        raise RuntimeError(f"No input PLY files found from {args.input_glob}")
    if args.runtime_log:
        args.runtime_log.parent.mkdir(parents=True, exist_ok=True)

    groups: dict[Path, list[Path]] = defaultdict(list)
    for path in files:
        groups[sequence_key(args.input_root, path)].append(path)

    distance_scale = distance_scale_from_paths(files, args.distance_unit)
    max_distance = args.max_distance * distance_scale if args.max_distance > 0 else None
    max_color_delta = args.max_color_delta if args.max_color_delta > 0 else None
    for _, seq_files in sorted(groups.items(), key=lambda item: str(item[0])):
        cache = {}
        for i, path in enumerate(seq_files):
            tic = time.time()
            out_path = args.out_root / relative_path(args.input_root, path)
            if args.skip_existing and out_path.exists():
                continue
            needed = [j for j in (i - 1, i, i + 1) if 0 <= j < len(seq_files)]
            for j in needed:
                if j not in cache:
                    xyz, rgb = read_ply_xyz_rgb(seq_files[j])
                    if rgb is None:
                        raise ValueError(f"{seq_files[j]} is missing RGB")
                    cache[j] = (xyz, rgb, cKDTree(xyz))
            for stale in list(cache):
                if stale < i - 1:
                    del cache[stale]

            cur = cache[i]
            prev_data = cache.get(i - 1)
            next_data = cache.get(i + 1)
            smoothed_rgb = smooth_frame(
                prev_data,
                cur,
                next_data,
                args.alpha,
                max_distance,
                max_color_delta,
                args.query_chunk_size,
            )
            write_ply_xyz_rgb(out_path, cur[0], smoothed_rgb)
            record = {
                "input": str(path),
                "output": str(out_path),
                "points": int(cur[0].shape[0]),
                "distance_unit": args.distance_unit,
                "resolved_max_distance": max_distance,
                "seconds": round(time.time() - tic, 4),
            }
            print(record)
            if args.runtime_log:
                with args.runtime_log.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(record) + "\n")


if __name__ == "__main__":
    main()
