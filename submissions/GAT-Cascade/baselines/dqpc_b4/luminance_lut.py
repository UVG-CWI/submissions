#!/usr/bin/env python3
"""Fit or apply a monotonic HE-supervised luminance correction LUT."""

from __future__ import annotations

import argparse
import glob
import json
import time
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from dqpc_data import distance_scale_from_paths, matching_he_path, read_ply_xyz_rgb, write_ply_xyz_rgb


def rgb_to_yuv(rgb: np.ndarray) -> np.ndarray:
    rgb = rgb.astype(np.float32)
    yuv = np.zeros_like(rgb, dtype=np.float32)
    yuv[:, 0] = 0.2126 * rgb[:, 0] + 0.7152 * rgb[:, 1] + 0.0722 * rgb[:, 2]
    yuv[:, 1] = -0.1146 * rgb[:, 0] - 0.3854 * rgb[:, 1] + 0.5000 * rgb[:, 2] + 128
    yuv[:, 2] = 0.5000 * rgb[:, 0] - 0.4542 * rgb[:, 1] - 0.0458 * rgb[:, 2] + 128
    return yuv


def yuv_to_rgb(yuv: np.ndarray) -> np.ndarray:
    centered = yuv.astype(np.float32).copy()
    centered[:, 1] -= 128
    centered[:, 2] -= 128
    rgb = np.zeros_like(centered, dtype=np.float32)
    rgb[:, 0] = centered[:, 0] + 1.57480 * centered[:, 2]
    rgb[:, 1] = centered[:, 0] - 0.18733 * centered[:, 1] - 0.46813 * centered[:, 2]
    rgb[:, 2] = centered[:, 0] + 1.85563 * centered[:, 1]
    return np.clip(np.rint(rgb), 0, 255)


def relative_path(root: Path, path: Path) -> Path:
    try:
        return path.resolve().relative_to(root.resolve())
    except ValueError:
        return Path(path.name)


def isotonic_increasing(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Weighted pool-adjacent-violators fit for a nondecreasing sequence."""
    blocks: list[list[float | int]] = []
    for index, (value, weight) in enumerate(zip(values, weights)):
        blocks.append([index, index + 1, float(value), max(float(weight), 1e-12)])
        while len(blocks) >= 2 and float(blocks[-2][2]) > float(blocks[-1][2]):
            left = blocks[-2]
            right = blocks[-1]
            merged_weight = float(left[3]) + float(right[3])
            merged_value = (
                float(left[2]) * float(left[3]) + float(right[2]) * float(right[3])
            ) / merged_weight
            blocks[-2:] = [[int(left[0]), int(right[1]), merged_value, merged_weight]]

    output = np.empty_like(values, dtype=np.float64)
    for start, end, value, _ in blocks:
        output[int(start) : int(end)] = float(value)
    return output


def frame_sample_indices(point_count: int, max_points: int, rng: np.random.Generator) -> np.ndarray:
    if max_points <= 0 or point_count <= max_points:
        return np.arange(point_count, dtype=np.int64)
    return np.sort(rng.choice(point_count, size=max_points, replace=False))


def fit_lut(args: argparse.Namespace) -> dict:
    source_paths = sorted(Path(p) for p in glob.glob(args.source_glob))
    if args.limit > 0:
        source_paths = source_paths[: args.limit]
    if not source_paths:
        raise RuntimeError(f"No source PLY files found from {args.source_glob}")

    bin_centers = np.linspace(0.0, 255.0, args.bins, dtype=np.float64)
    target_sum = np.zeros(args.bins, dtype=np.float64)
    target_count = np.zeros(args.bins, dtype=np.float64)
    rng = np.random.default_rng(args.seed)
    distance_scale = distance_scale_from_paths(source_paths, args.distance_unit)
    max_distance = args.max_distance * distance_scale
    matched_points = 0
    sampled_points = 0

    for frame_idx, source_path in enumerate(source_paths, start=1):
        rel = relative_path(args.source_root, source_path)
        gt_path = matching_he_path(args.gt_root, rel) or (args.gt_root / rel)
        if not gt_path.exists():
            raise FileNotFoundError(f"Missing GT for {source_path}: {gt_path}")

        source_xyz, source_rgb = read_ply_xyz_rgb(source_path)
        gt_xyz, gt_rgb = read_ply_xyz_rgb(gt_path)
        if source_rgb is None:
            raise ValueError(f"{source_path} is missing RGB")
        if gt_rgb is None:
            raise ValueError(f"{gt_path} is missing RGB")

        sample_idx = frame_sample_indices(source_xyz.shape[0], args.max_points_per_frame, rng)
        sampled_xyz = source_xyz[sample_idx]
        sampled_y = rgb_to_yuv(source_rgb[sample_idx])[:, 0].astype(np.float64)
        dist, gt_idx = cKDTree(gt_xyz).query(sampled_xyz, k=1, workers=-1)
        target_y = rgb_to_yuv(gt_rgb[gt_idx])[:, 0].astype(np.float64)

        valid = np.isfinite(dist) & np.isfinite(sampled_y) & np.isfinite(target_y)
        if args.max_distance > 0:
            valid &= dist <= max_distance
        if args.max_target_delta > 0:
            valid &= np.abs(target_y - sampled_y) <= args.max_target_delta
        if not np.any(valid):
            print(f"[{frame_idx}/{len(source_paths)}] no reliable matches: {rel}")
            continue

        source_valid = sampled_y[valid]
        target_valid = target_y[valid]
        bin_idx = np.clip(
            np.rint(source_valid * (args.bins - 1) / 255.0).astype(np.int64),
            0,
            args.bins - 1,
        )
        target_sum += np.bincount(bin_idx, weights=target_valid, minlength=args.bins)
        target_count += np.bincount(bin_idx, minlength=args.bins)
        sampled_points += int(sample_idx.shape[0])
        matched_points += int(np.sum(valid))
        print(
            f"[{frame_idx}/{len(source_paths)}] {rel} "
            f"sampled={sample_idx.shape[0]} reliable={int(np.sum(valid))}"
        )

    if matched_points == 0:
        raise RuntimeError("No reliable train-to-HE luminance correspondences were found")

    weighted_sum = target_sum + args.identity_prior * bin_centers
    weighted_count = target_count + args.identity_prior
    raw_lut = weighted_sum / np.maximum(weighted_count, 1e-12)
    lut = isotonic_increasing(raw_lut, weighted_count)
    if args.max_correction > 0:
        lut = np.clip(
            lut,
            bin_centers - args.max_correction,
            bin_centers + args.max_correction,
        )
    lut = np.clip(lut, 0.0, 255.0)

    return {
        "model_type": "monotonic_luminance_lut",
        "bins": args.bins,
        "bin_centers": bin_centers.tolist(),
        "lut": lut.tolist(),
        "raw_lut": raw_lut.tolist(),
        "bin_match_count": target_count.astype(int).tolist(),
        "source_frame_count": len(source_paths),
        "sampled_point_count": sampled_points,
        "reliable_match_count": matched_points,
        "reliable_match_ratio": matched_points / max(sampled_points, 1),
        "max_distance": args.max_distance,
        "distance_unit": args.distance_unit,
        "resolved_max_distance": max_distance,
        "max_target_delta": args.max_target_delta,
        "identity_prior": args.identity_prior,
        "max_correction": args.max_correction,
        "seed": args.seed,
    }


def apply_lut(args: argparse.Namespace, model: dict) -> None:
    source_paths = sorted(Path(p) for p in glob.glob(args.source_glob))
    if args.limit > 0:
        source_paths = source_paths[: args.limit]
    if not source_paths:
        raise RuntimeError(f"No source PLY files found from {args.source_glob}")

    bin_centers = np.asarray(model["bin_centers"], dtype=np.float64)
    lut = np.asarray(model["lut"], dtype=np.float64)
    if bin_centers.ndim != 1 or lut.shape != bin_centers.shape:
        raise ValueError("Invalid LUT model arrays")
    if not 0.0 <= args.strength <= 1.0:
        raise ValueError("--strength must be in [0, 1]")
    if args.runtime_log:
        args.runtime_log.parent.mkdir(parents=True, exist_ok=True)

    for frame_idx, source_path in enumerate(source_paths, start=1):
        tic = time.time()
        rel = relative_path(args.source_root, source_path)
        out_path = args.out_root / rel
        if args.skip_existing and out_path.exists():
            print(f"[{frame_idx}/{len(source_paths)}] skip existing {out_path}")
            continue

        xyz, rgb = read_ply_xyz_rgb(source_path)
        if rgb is None:
            raise ValueError(f"{source_path} is missing RGB")
        yuv = rgb_to_yuv(rgb)
        mapped_y = np.interp(yuv[:, 0], bin_centers, lut)
        delta = mapped_y - yuv[:, 0]
        if args.max_delta > 0:
            delta = np.clip(delta, -args.max_delta, args.max_delta)
        yuv[:, 0] = np.clip(yuv[:, 0] + args.strength * delta, 0.0, 255.0)
        out_rgb = yuv_to_rgb(yuv)
        write_ply_xyz_rgb(out_path, xyz, out_rgb)

        record = {
            "input": str(source_path),
            "output": str(out_path),
            "points": int(xyz.shape[0]),
            "strength": args.strength,
            "mean_abs_y_delta": float(np.mean(np.abs(args.strength * delta))),
            "max_abs_y_delta": float(np.max(np.abs(args.strength * delta))),
            "seconds": round(time.time() - tic, 4),
        }
        print(json.dumps(record))
        if args.runtime_log:
            with args.runtime_log.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="HE-supervised luminance LUT")
    parser.add_argument("--mode", choices=("fit", "apply"), required=True)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--source-glob", required=True)
    parser.add_argument("--gt-root", default=None, type=Path)
    parser.add_argument("--model-in", default=None, type=Path)
    parser.add_argument("--model-out", default=None, type=Path)
    parser.add_argument("--out-root", default=None, type=Path)
    parser.add_argument("--runtime-log", default=None, type=Path)
    parser.add_argument("--bins", default=256, type=int)
    parser.add_argument("--max-points-per-frame", default=200000, type=int)
    parser.add_argument("--max-distance", default=10.0, type=float)
    parser.add_argument("--distance-unit", choices=("mm", "coordinate"), default="mm")
    parser.add_argument("--max-target-delta", default=80.0, type=float)
    parser.add_argument("--identity-prior", default=500.0, type=float)
    parser.add_argument("--max-correction", default=40.0, type=float)
    parser.add_argument("--strength", default=1.0, type=float)
    parser.add_argument("--max-delta", default=40.0, type=float)
    parser.add_argument("--seed", default=1, type=int)
    parser.add_argument("--limit", default=0, type=int)
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "fit":
        if args.gt_root is None:
            raise ValueError("--gt-root is required in fit mode")
        if args.model_out is None:
            raise ValueError("--model-out is required in fit mode")
        if args.bins < 2:
            raise ValueError("--bins must be at least 2")
        model = fit_lut(args)
        args.model_out.parent.mkdir(parents=True, exist_ok=True)
        args.model_out.write_text(json.dumps(model, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({key: value for key, value in model.items() if not isinstance(value, list)}, indent=2))
        return

    if args.model_in is None:
        raise ValueError("--model-in is required in apply mode")
    if args.out_root is None:
        raise ValueError("--out-root is required in apply mode")
    model = json.loads(args.model_in.read_text(encoding="utf-8"))
    apply_lut(args, model)


if __name__ == "__main__":
    main()
