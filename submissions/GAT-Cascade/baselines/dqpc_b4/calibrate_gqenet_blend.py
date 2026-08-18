#!/usr/bin/env python3
"""Calibrate and apply conservative YUV blending for GQE-Net outputs."""

from __future__ import annotations

import argparse
import glob
import json
import time
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from dqpc_data import matching_he_path, read_ply_xyz_rgb, write_ply_xyz_rgb


CHANNEL_NAMES = ("y", "u", "v")


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


def psnr_from_mse(mse: np.ndarray | float) -> np.ndarray | float:
    mse_array = np.asarray(mse, dtype=np.float64)
    safe = np.maximum(mse_array, 1e-12)
    result = 10.0 * np.log10((255.0 * 255.0) / safe)
    if np.ndim(mse) == 0:
        return float(result)
    return result


def validate_pair(
    base_path: Path,
    candidate_path: Path,
    xyz_atol: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    base_xyz, base_rgb = read_ply_xyz_rgb(base_path)
    candidate_xyz, candidate_rgb = read_ply_xyz_rgb(candidate_path)
    if base_rgb is None:
        raise ValueError(f"{base_path} is missing RGB")
    if candidate_rgb is None:
        raise ValueError(f"{candidate_path} is missing RGB")
    if base_xyz.shape != candidate_xyz.shape:
        raise ValueError(
            f"GQE candidate changed point count for {base_path}: "
            f"{base_xyz.shape[0]} vs {candidate_xyz.shape[0]}"
        )
    if not np.allclose(base_xyz, candidate_xyz, rtol=0.0, atol=xyz_atol):
        raise ValueError(f"GQE candidate changed XYZ or point order: {candidate_path}")
    return candidate_xyz, base_rgb, candidate_rgb


def calibrate(
    base_paths: list[Path],
    base_root: Path,
    candidate_root: Path,
    gt_root: Path,
    steps: int,
    xyz_atol: float,
) -> dict:
    alphas = np.linspace(0.0, 1.0, steps, dtype=np.float64)
    psnr_sums = np.zeros((3, steps), dtype=np.float64)
    frame_count = 0

    for frame_idx, base_path in enumerate(base_paths, start=1):
        rel = relative_path(base_root, base_path)
        candidate_path = candidate_root / rel
        gt_path = matching_he_path(gt_root, rel) or (gt_root / rel)
        if not candidate_path.exists():
            raise FileNotFoundError(f"Missing GQE candidate for {base_path}: {candidate_path}")
        if not gt_path.exists():
            raise FileNotFoundError(f"Missing GT for {base_path}: {gt_path}")

        xyz, base_rgb, candidate_rgb = validate_pair(base_path, candidate_path, xyz_atol)
        gt_xyz, gt_rgb = read_ply_xyz_rgb(gt_path)
        if gt_rgb is None:
            raise ValueError(f"{gt_path} is missing RGB")
        _, gt_idx = cKDTree(gt_xyz).query(xyz, k=1, workers=-1)

        base_yuv = rgb_to_yuv(base_rgb).astype(np.float64)
        candidate_yuv = rgb_to_yuv(candidate_rgb).astype(np.float64)
        target_yuv = rgb_to_yuv(gt_rgb[gt_idx]).astype(np.float64)
        error = base_yuv - target_yuv
        delta = candidate_yuv - base_yuv

        for channel in range(3):
            a = float(np.mean(delta[:, channel] ** 2))
            b = float(np.mean(error[:, channel] * delta[:, channel]))
            c = float(np.mean(error[:, channel] ** 2))
            mse = c + 2.0 * alphas * b + (alphas**2) * a
            psnr_sums[channel] += psnr_from_mse(mse)
        frame_count += 1
        print(f"[{frame_idx}/{len(base_paths)}] calibrated {rel}")

    mean_psnr = psnr_sums / max(frame_count, 1)
    selected_indices = np.argmax(mean_psnr, axis=1)
    blends = [float(alphas[index]) for index in selected_indices]
    config = {
        "blend_y": blends[0],
        "blend_u": blends[1],
        "blend_v": blends[2],
        "calibrated_blend_y": blends[0],
        "calibrated_blend_u": blends[1],
        "calibrated_blend_v": blends[2],
        "max_delta_y": 0.0,
        "max_delta_u": 0.0,
        "max_delta_v": 0.0,
        "calibration_frames": frame_count,
        "calibration_steps": steps,
        "baseline_mean_psnr_y": float(mean_psnr[0, 0]),
        "baseline_mean_psnr_u": float(mean_psnr[1, 0]),
        "baseline_mean_psnr_v": float(mean_psnr[2, 0]),
        "candidate_mean_psnr_y": float(mean_psnr[0, -1]),
        "candidate_mean_psnr_u": float(mean_psnr[1, -1]),
        "candidate_mean_psnr_v": float(mean_psnr[2, -1]),
        "selected_mean_psnr_y": float(mean_psnr[0, selected_indices[0]]),
        "selected_mean_psnr_u": float(mean_psnr[1, selected_indices[1]]),
        "selected_mean_psnr_v": float(mean_psnr[2, selected_indices[2]]),
    }
    return config


def config_arrays(config: dict) -> tuple[np.ndarray, np.ndarray]:
    blends = np.asarray([config[f"blend_{name}"] for name in CHANNEL_NAMES], dtype=np.float32)
    max_deltas = np.asarray(
        [float(config.get(f"max_delta_{name}", 0.0)) for name in CHANNEL_NAMES],
        dtype=np.float32,
    )
    if np.any(blends < 0) or np.any(blends > 1):
        raise ValueError(f"Blend values must be in [0, 1], got {blends.tolist()}")
    return blends, max_deltas


def blend_rgb(base_rgb: np.ndarray, candidate_rgb: np.ndarray, config: dict) -> np.ndarray:
    blends, max_deltas = config_arrays(config)
    if np.all(blends == 0):
        return base_rgb
    base_yuv = rgb_to_yuv(base_rgb)
    candidate_yuv = rgb_to_yuv(candidate_rgb)
    delta = candidate_yuv - base_yuv
    enabled_caps = max_deltas > 0
    if np.any(enabled_caps):
        delta[:, enabled_caps] = np.clip(
            delta[:, enabled_caps],
            -max_deltas[enabled_caps],
            max_deltas[enabled_caps],
        )
    out_yuv = np.clip(base_yuv + delta * blends[None, :], 0.0, 255.0)
    return yuv_to_rgb(out_yuv)


def actual_mean_psnr(
    base_paths: list[Path],
    base_root: Path,
    candidate_root: Path,
    gt_root: Path,
    config: dict,
    xyz_atol: float,
) -> tuple[np.ndarray, np.ndarray]:
    base_psnr_sum = np.zeros(3, dtype=np.float64)
    output_psnr_sum = np.zeros(3, dtype=np.float64)
    for base_path in base_paths:
        rel = relative_path(base_root, base_path)
        candidate_path = candidate_root / rel
        gt_path = matching_he_path(gt_root, rel) or (gt_root / rel)
        xyz, base_rgb, candidate_rgb = validate_pair(base_path, candidate_path, xyz_atol)
        gt_xyz, gt_rgb = read_ply_xyz_rgb(gt_path)
        if gt_rgb is None:
            raise ValueError(f"{gt_path} is missing RGB")
        _, gt_idx = cKDTree(gt_xyz).query(xyz, k=1, workers=-1)
        target_yuv = rgb_to_yuv(gt_rgb[gt_idx]).astype(np.float64)
        base_yuv = rgb_to_yuv(base_rgb).astype(np.float64)
        output_yuv = rgb_to_yuv(blend_rgb(base_rgb, candidate_rgb, config)).astype(np.float64)
        for channel in range(3):
            base_mse = float(np.mean((base_yuv[:, channel] - target_yuv[:, channel]) ** 2))
            output_mse = float(np.mean((output_yuv[:, channel] - target_yuv[:, channel]) ** 2))
            base_psnr_sum[channel] += psnr_from_mse(base_mse)
            output_psnr_sum[channel] += psnr_from_mse(output_mse)
    count = max(len(base_paths), 1)
    return base_psnr_sum / count, output_psnr_sum / count


def enforce_actual_non_regression(
    base_paths: list[Path],
    base_root: Path,
    candidate_root: Path,
    gt_root: Path,
    config: dict,
    xyz_atol: float,
    minimum_gains: dict[str, float] | None = None,
) -> dict:
    minimum_gains = minimum_gains or {name: 0.0 for name in CHANNEL_NAMES}
    base_psnr, output_psnr = actual_mean_psnr(
        base_paths,
        base_root,
        candidate_root,
        gt_root,
        config,
        xyz_atol,
    )
    proposed_output_psnr = output_psnr.copy()
    worse = [
        channel
        for channel in range(3)
        if output_psnr[channel] + 1e-9 < base_psnr[channel]
    ]
    insufficient_gain = [
        channel
        for channel, name in enumerate(CHANNEL_NAMES)
        if output_psnr[channel] + 1e-9
        < base_psnr[channel] + max(float(minimum_gains[name]), 0.0)
    ]
    fallback_channels = [CHANNEL_NAMES[channel] for channel in worse]
    insufficient_gain_channels = [CHANNEL_NAMES[channel] for channel in insufficient_gain]
    if insufficient_gain:
        # RGB clipping couples Y/U/V. Falling back all channels is the only
        # strict way to preserve every baseline channel without another sweep.
        config.update({"blend_y": 0.0, "blend_u": 0.0, "blend_v": 0.0})
        output_psnr = base_psnr.copy()
    for channel, name in enumerate(CHANNEL_NAMES):
        config[f"applied_baseline_mean_psnr_{name}"] = float(base_psnr[channel])
        config[f"proposed_mean_psnr_{name}"] = float(proposed_output_psnr[channel])
        config[f"applied_selected_mean_psnr_{name}"] = float(output_psnr[channel])
    config["non_regression_fallback_channels"] = fallback_channels
    config["minimum_required_gain_db"] = {
        name: max(float(minimum_gains[name]), 0.0) for name in CHANNEL_NAMES
    }
    config["insufficient_gain_fallback_channels"] = insufficient_gain_channels
    config["candidate_used"] = any(float(config[f"blend_{name}"]) > 0 for name in CHANNEL_NAMES)
    config["selection_decision"] = "candidate_blend" if config["candidate_used"] else "base_fallback"
    return config


def add_quality_gate(config: dict, minimums: dict[str, float]) -> dict:
    failures = []
    for name in CHANNEL_NAMES:
        threshold = float(minimums[name])
        value = config.get(f"applied_selected_mean_psnr_{name}")
        if threshold > 0 and (not isinstance(value, (int, float)) or float(value) < threshold):
            failures.append(
                {
                    "channel": name,
                    "required": threshold,
                    "actual": float(value) if isinstance(value, (int, float)) else None,
                }
            )
    config["quality_gate_min_psnr"] = minimums
    config["quality_gate_failures"] = failures
    config["quality_gate_passed"] = not failures
    return config


def color_drift_stats(base_rgb: np.ndarray, output_rgb: np.ndarray) -> dict:
    base_yuv = rgb_to_yuv(base_rgb).astype(np.float64)
    output_yuv = rgb_to_yuv(output_rgb).astype(np.float64)
    abs_delta = np.abs(output_yuv - base_yuv)
    base_clip = np.mean((base_rgb <= 1) | (base_rgb >= 254), axis=0)
    output_clip = np.mean((output_rgb <= 1) | (output_rgb >= 254), axis=0)
    return {
        "mean_abs_yuv": np.mean(abs_delta, axis=0),
        "p99_abs_yuv": np.percentile(abs_delta, 99, axis=0),
        "rgb_clip_fraction_increase": np.maximum(output_clip - base_clip, 0.0),
    }


def add_color_drift_guard(
    base_paths: list[Path],
    base_root: Path,
    candidate_root: Path,
    config: dict,
    xyz_atol: float,
) -> dict:
    frame_stats = []
    for base_path in base_paths:
        rel = relative_path(base_root, base_path)
        candidate_path = candidate_root / rel
        _, base_rgb, candidate_rgb = validate_pair(base_path, candidate_path, xyz_atol)
        output_rgb = blend_rgb(base_rgb, candidate_rgb, config)
        frame_stats.append(color_drift_stats(base_rgb, output_rgb))

    guard = {
        "enabled": True,
        "profile_frames": len(frame_stats),
        "fallback": "base",
    }
    for channel, name in enumerate(CHANNEL_NAMES):
        max_mean = max(float(stats["mean_abs_yuv"][channel]) for stats in frame_stats)
        max_p99 = max(float(stats["p99_abs_yuv"][channel]) for stats in frame_stats)
        guard[f"max_mean_abs_{name}"] = max(1.0, 2.0 * max_mean + 0.5)
        guard[f"max_p99_abs_{name}"] = max(3.0, 1.5 * max_p99 + 2.0)
    for channel, name in enumerate(("r", "g", "b")):
        max_clip_increase = max(
            float(stats["rgb_clip_fraction_increase"][channel]) for stats in frame_stats
        )
        guard[f"max_clip_fraction_increase_{name}"] = max(0.02, 2.0 * max_clip_increase + 0.01)
    config["color_drift_guard"] = guard
    return config


def color_drift_failures(base_rgb: np.ndarray, output_rgb: np.ndarray, config: dict) -> list[dict]:
    guard = config.get("color_drift_guard")
    if not isinstance(guard, dict) or not bool(guard.get("enabled", False)):
        return []
    stats = color_drift_stats(base_rgb, output_rgb)
    failures = []
    for channel, name in enumerate(CHANNEL_NAMES):
        for stat_name, key_prefix in (
            ("mean_abs_yuv", "max_mean_abs"),
            ("p99_abs_yuv", "max_p99_abs"),
        ):
            threshold = guard.get(f"{key_prefix}_{name}")
            value = float(stats[stat_name][channel])
            if isinstance(threshold, (int, float)) and value > float(threshold):
                failures.append(
                    {
                        "metric": f"{stat_name}_{name}",
                        "actual": value,
                        "maximum": float(threshold),
                    }
                )
    for channel, name in enumerate(("r", "g", "b")):
        threshold = guard.get(f"max_clip_fraction_increase_{name}")
        value = float(stats["rgb_clip_fraction_increase"][channel])
        if isinstance(threshold, (int, float)) and value > float(threshold):
            failures.append(
                {
                    "metric": f"rgb_clip_fraction_increase_{name}",
                    "actual": value,
                    "maximum": float(threshold),
                }
            )
    return failures


def apply_blend(
    base_paths: list[Path],
    base_root: Path,
    candidate_root: Path,
    out_root: Path,
    config: dict,
    xyz_atol: float,
    runtime_log: Path | None,
) -> None:
    blends, _ = config_arrays(config)
    if runtime_log:
        runtime_log.parent.mkdir(parents=True, exist_ok=True)

    for frame_idx, base_path in enumerate(base_paths, start=1):
        tic = time.time()
        rel = relative_path(base_root, base_path)
        candidate_path = candidate_root / rel
        out_path = out_root / rel
        if not candidate_path.exists():
            raise FileNotFoundError(f"Missing GQE candidate for {base_path}: {candidate_path}")
        xyz, base_rgb, candidate_rgb = validate_pair(base_path, candidate_path, xyz_atol)
        out_rgb = blend_rgb(base_rgb, candidate_rgb, config)
        drift_failures = color_drift_failures(base_rgb, out_rgb, config)
        if drift_failures:
            out_rgb = base_rgb
        write_ply_xyz_rgb(out_path, xyz, out_rgb)

        record = {
            "base": str(base_path),
            "candidate": str(candidate_path),
            "output": str(out_path),
            "points": int(xyz.shape[0]),
            "blend_y": float(blends[0]),
            "blend_u": float(blends[1]),
            "blend_v": float(blends[2]),
            "color_drift_guard_fallback": bool(drift_failures),
            "color_drift_guard_failures": drift_failures,
            "seconds": round(time.time() - tic, 4),
        }
        print(json.dumps(record))
        if runtime_log:
            with runtime_log.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate/apply GQE-Net YUV blend weights")
    parser.add_argument("--base-root", required=True, type=Path)
    parser.add_argument("--base-glob", required=True)
    parser.add_argument("--candidate-root", required=True, type=Path)
    parser.add_argument("--gt-root", default=None, type=Path)
    parser.add_argument("--config-in", default=None, type=Path)
    parser.add_argument("--config-out", default=None, type=Path)
    parser.add_argument("--out-root", default=None, type=Path)
    parser.add_argument("--runtime-log", default=None, type=Path)
    parser.add_argument("--steps", default=101, type=int)
    parser.add_argument("--xyz-atol", default=1e-4, type=float)
    parser.add_argument("--min-psnr-y", default=0.0, type=float)
    parser.add_argument("--min-psnr-u", default=0.0, type=float)
    parser.add_argument("--min-psnr-v", default=0.0, type=float)
    parser.add_argument("--min-gain-y", default=0.0, type=float)
    parser.add_argument("--min-gain-u", default=0.0, type=float)
    parser.add_argument("--min-gain-v", default=0.0, type=float)
    parser.add_argument("--require-quality-gate", action="store_true")
    parser.add_argument("--require-color-drift-guard", action="store_true")
    parser.add_argument("--limit", default=0, type=int)
    args = parser.parse_args()

    if args.steps < 2:
        raise ValueError("--steps must be at least 2")
    base_paths = sorted(Path(p) for p in glob.glob(args.base_glob))
    if args.limit > 0:
        base_paths = base_paths[: args.limit]
    if not base_paths:
        raise RuntimeError(f"No base PLY files found from {args.base_glob}")

    if args.config_in:
        config = json.loads(args.config_in.read_text(encoding="utf-8"))
    else:
        if args.gt_root is None:
            raise ValueError("--gt-root is required when --config-in is not provided")
        config = calibrate(
            base_paths,
            args.base_root,
            args.candidate_root,
            args.gt_root,
            args.steps,
            args.xyz_atol,
        )
        config = enforce_actual_non_regression(
            base_paths,
            args.base_root,
            args.candidate_root,
            args.gt_root,
            config,
            args.xyz_atol,
            {
                "y": args.min_gain_y,
                "u": args.min_gain_u,
                "v": args.min_gain_v,
            },
        )
        config = add_color_drift_guard(
            base_paths,
            args.base_root,
            args.candidate_root,
            config,
            args.xyz_atol,
        )
        config = add_quality_gate(
            config,
            {
                "y": args.min_psnr_y,
                "u": args.min_psnr_u,
                "v": args.min_psnr_v,
            },
        )

    print(json.dumps(config, indent=2))
    if args.config_out:
        args.config_out.parent.mkdir(parents=True, exist_ok=True)
        args.config_out.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    if args.require_quality_gate and not bool(config.get("quality_gate_passed", False)):
        raise RuntimeError(
            "PSNR quality gate failed; this run did not write final output. "
            "Any pre-existing output directory must be treated as stale. "
            f"Failures: {config.get('quality_gate_failures', [])}"
        )
    drift_guard = config.get("color_drift_guard")
    if args.require_color_drift_guard and (
        not isinstance(drift_guard, dict) or not bool(drift_guard.get("enabled", False))
    ):
        raise RuntimeError(
            "Color drift guard is required but missing from the blend config. "
            "Regenerate the validation config with the current calibration code."
        )
    if args.out_root:
        apply_blend(
            base_paths,
            args.base_root,
            args.candidate_root,
            args.out_root,
            config,
            args.xyz_atol,
            args.runtime_log,
        )


if __name__ == "__main__":
    main()
