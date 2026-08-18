#!/usr/bin/env python3
"""Basic local evaluation: NN geometry distance, F-score, and YUV PSNR."""

from __future__ import annotations

import argparse
import glob
import json
import math
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from dqpc_data import distance_scale_from_paths, matching_he_path, read_ply_xyz_rgb


def relative_path(root: Path, path: Path) -> Path:
    try:
        return path.resolve().relative_to(root.resolve())
    except ValueError:
        return Path(path.name)


def rgb_to_yuv(rgb: np.ndarray) -> np.ndarray:
    rgb = rgb.astype(np.float32)
    yuv = np.zeros_like(rgb, dtype=np.float32)
    yuv[:, 0] = 0.2126 * rgb[:, 0] + 0.7152 * rgb[:, 1] + 0.0722 * rgb[:, 2]
    yuv[:, 1] = -0.1146 * rgb[:, 0] - 0.3854 * rgb[:, 1] + 0.5000 * rgb[:, 2] + 128
    yuv[:, 2] = 0.5000 * rgb[:, 0] - 0.4542 * rgb[:, 1] - 0.0458 * rgb[:, 2] + 128
    return yuv


def psnr(a: np.ndarray, b: np.ndarray, peak: float = 255.0) -> float:
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    if mse <= 0:
        return float("inf")
    return 20.0 * math.log10(peak / math.sqrt(mse))


def parse_thresholds(text: str) -> list[float]:
    thresholds = []
    for item in text.split(","):
        item = item.strip()
        if item:
            thresholds.append(float(item))
    if not thresholds:
        raise ValueError("At least one F-score threshold is required")
    return sorted(set(thresholds))


def metric_suffix(value: float) -> str:
    text = f"{value:g}".replace("-", "m").replace(".", "p")
    return text


def fscore_from_distances(d_pred_gt: np.ndarray, d_gt_pred: np.ndarray, threshold: float) -> tuple[float, float, float]:
    precision = float(np.mean(d_pred_gt <= threshold))
    recall = float(np.mean(d_gt_pred <= threshold))
    if precision + recall <= 0:
        fscore = 0.0
    else:
        fscore = 2.0 * precision * recall / (precision + recall)
    return precision, recall, fscore


def bbox_diagnostics(pred_xyz: np.ndarray, gt_xyz: np.ndarray) -> dict:
    gt_min = gt_xyz.min(axis=0)
    gt_max = gt_xyz.max(axis=0)
    pred_min = pred_xyz.min(axis=0)
    pred_max = pred_xyz.max(axis=0)
    gt_extent = gt_max - gt_min
    pred_extent = pred_max - pred_min
    gt_diag = float(np.linalg.norm(gt_extent))
    pred_diag = float(np.linalg.norm(pred_extent))
    centroid_delta = float(np.linalg.norm(pred_xyz.mean(axis=0) - gt_xyz.mean(axis=0)))
    return {
        "gt_bbox_diag": gt_diag,
        "pred_bbox_diag": pred_diag,
        "bbox_diag_ratio": pred_diag / gt_diag if gt_diag > 0 else float("nan"),
        "centroid_delta": centroid_delta,
    }


def eval_pair(
    pred_path: Path,
    gt_path: Path,
    fscore_thresholds: list[tuple[float, float]],
    threshold_unit: str = "coordinate",
) -> dict:
    pred_xyz, pred_rgb = read_ply_xyz_rgb(pred_path)
    gt_xyz, gt_rgb = read_ply_xyz_rgb(gt_path)
    tree_gt = cKDTree(gt_xyz)
    d_pred_gt, idx_gt = tree_gt.query(pred_xyz, k=1, workers=-1)
    tree_pred = cKDTree(pred_xyz)
    d_gt_pred, _ = tree_pred.query(gt_xyz, k=1, workers=-1)
    pred_to_gt_mse = float(np.mean(d_pred_gt**2))
    gt_to_pred_mse = float(np.mean(d_gt_pred**2))
    pred_to_gt_mean = float(np.mean(d_pred_gt))
    gt_to_pred_mean = float(np.mean(d_gt_pred))
    chamfer_l2 = pred_to_gt_mse + gt_to_pred_mse
    symmetric_mse = 0.5 * chamfer_l2
    frame_symmetric_rmse = math.sqrt(symmetric_mse)
    result = {
        "pred": str(pred_path),
        "gt": str(gt_path),
        "pred_points": int(pred_xyz.shape[0]),
        "gt_points": int(gt_xyz.shape[0]),
        "point_count_ratio": float(pred_xyz.shape[0] / gt_xyz.shape[0]) if gt_xyz.shape[0] else float("nan"),
        "chamfer_l2_sum_mse": chamfer_l2,
        "symmetric_mse_mean": symmetric_mse,
        "frame_symmetric_rmse": frame_symmetric_rmse,
        # Legacy aliases retained for existing result parsers.
        "cd_mse_approx": chamfer_l2,
        "cd_mse_mean": symmetric_mse,
        "cd_rmse_symmetric": frame_symmetric_rmse,
        "pred_to_gt_mse": pred_to_gt_mse,
        "gt_to_pred_mse": gt_to_pred_mse,
        "pred_to_gt_rmse": float(math.sqrt(pred_to_gt_mse)),
        "gt_to_pred_rmse": float(math.sqrt(gt_to_pred_mse)),
        "CD_Acc": pred_to_gt_mean,
        "CD_Comp": gt_to_pred_mean,
        "chamfer-L1": pred_to_gt_mean + gt_to_pred_mean,
        "chamfer-L2": chamfer_l2,
        "chamferL2_old": 0.5 * (pred_to_gt_mean + gt_to_pred_mean),
        "fscore_threshold_unit": threshold_unit,
    }
    result.update(bbox_diagnostics(pred_xyz, gt_xyz))
    gt_diag = float(result["gt_bbox_diag"])
    if gt_diag > 0:
        result["cd_rmse_symmetric_norm_bbox"] = result["cd_rmse_symmetric"] / gt_diag
        result["pred_to_gt_rmse_norm_bbox"] = result["pred_to_gt_rmse"] / gt_diag
        result["gt_to_pred_rmse_norm_bbox"] = result["gt_to_pred_rmse"] / gt_diag
        result["centroid_delta_norm_bbox"] = result["centroid_delta"] / gt_diag
    for reported_threshold, resolved_threshold in fscore_thresholds:
        precision, recall, fscore = fscore_from_distances(d_pred_gt, d_gt_pred, resolved_threshold)
        suffix = metric_suffix(reported_threshold)
        result[f"precision_tau_{suffix}"] = precision
        result[f"recall_tau_{suffix}"] = recall
        result[f"fscore_tau_{suffix}"] = fscore
        result[f"P_{reported_threshold:g}"] = precision
        result[f"R_{reported_threshold:g}"] = recall
        result[f"F_{reported_threshold:g}"] = fscore
    if pred_rgb is not None and gt_rgb is not None:
        gt_yuv = rgb_to_yuv(gt_rgb[idx_gt])
        pred_yuv = rgb_to_yuv(pred_rgb)
        result["psnr_y"] = psnr(pred_yuv[:, 0], gt_yuv[:, 0])
        result["psnr_u"] = psnr(pred_yuv[:, 1], gt_yuv[:, 1])
        result["psnr_v"] = psnr(pred_yuv[:, 2], gt_yuv[:, 2])
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Basic DQPC evaluation")
    parser.add_argument("--pred-root", required=True, type=Path)
    parser.add_argument("--pred-glob", required=True)
    parser.add_argument("--gt-root", required=True, type=Path)
    parser.add_argument("--out-jsonl", required=True, type=Path)
    parser.add_argument("--summary-json", default=None, type=Path)
    parser.add_argument(
        "--fscore-thresholds",
        default="5,10,20,30",
        help="Comma-separated distance thresholds in the PLY coordinate unit.",
    )
    parser.add_argument("--limit", default=0, type=int)
    parser.add_argument("--threshold-unit", choices=("mm", "coordinate"), default="mm")
    args = parser.parse_args()
    requested_thresholds = parse_thresholds(args.fscore_thresholds)

    preds = sorted(Path(p) for p in glob.glob(args.pred_glob))
    if args.limit > 0:
        preds = preds[: args.limit]
    if not preds:
        raise RuntimeError(f"No prediction files found from {args.pred_glob}")
    distance_scale = distance_scale_from_paths(preds, args.threshold_unit)
    fscore_thresholds = [
        (threshold, threshold * distance_scale)
        for threshold in requested_thresholds
    ]
    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    records = []
    with args.out_jsonl.open("w", encoding="utf-8") as f:
        for pred in preds:
            rel = relative_path(args.pred_root, pred)
            gt = matching_he_path(args.gt_root, rel) or (args.gt_root / rel)
            if not gt.exists():
                raise FileNotFoundError(f"Missing GT for {pred}: {gt}")
            record = eval_pair(pred, gt, fscore_thresholds, args.threshold_unit)
            records.append(record)
            f.write(json.dumps(record) + "\n")
            print(record)

    numeric_keys = sorted(
        {
            key
            for record in records
            for key, value in record.items()
            if isinstance(value, (int, float))
        }
    )
    summary = {
        "count": len(records),
        "metric_definitions": {
            "mean_chamfer-L2": "Official UVG-CWI Metric: mean(pred_to_gt squared distance) + mean(gt_to_pred squared distance).",
            "mean_symmetric_mse_mean": "Diagnostic only: 0.5 * chamfer-L2, computed per frame then averaged.",
            "mean_frame_symmetric_rmse": "Diagnostic only: per-frame sqrt(0.5 * chamfer-L2), then averaged across frames.",
            "rmse_from_mean_symmetric_mse": "Diagnostic only: sqrt(mean symmetric MSE across frames); differs from mean frame RMSE.",
        },
    }
    for key in numeric_keys:
        vals = [r[key] for r in records if key in r and math.isfinite(float(r[key]))]
        if vals:
            summary[f"mean_{key}"] = float(np.mean(vals))
    mean_chamfer_l2 = summary.get("mean_chamfer-L2")
    mean_symmetric_mse = summary.get("mean_symmetric_mse_mean")
    mean_frame_rmse = summary.get("mean_frame_symmetric_rmse")
    if isinstance(mean_symmetric_mse, (int, float)):
        summary["rmse_from_mean_symmetric_mse"] = float(math.sqrt(mean_symmetric_mse))
    if isinstance(mean_chamfer_l2, (int, float)) and isinstance(mean_symmetric_mse, (int, float)):
        summary["chamfer_l2_identity_error"] = float(abs(mean_chamfer_l2 - 2.0 * mean_symmetric_mse))
    if isinstance(mean_frame_rmse, (int, float)) and "rmse_from_mean_symmetric_mse" in summary:
        summary["rmse_aggregation_gap"] = float(
            summary["rmse_from_mean_symmetric_mse"] - mean_frame_rmse
        )
    print(json.dumps(summary, indent=2))
    if args.summary_json:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
