#!/usr/bin/env python3
"""Collect DQPC validation metrics into one quality-first board."""

from __future__ import annotations

import argparse
import glob
import json
import math
from pathlib import Path


BOARD_KEYS = (
    "uvg_metric_count",
    "mean_CD_Acc",
    "mean_CD_Comp",
    "mean_chamfer-L1",
    "mean_chamfer-L2",
    "mean_derived_symmetric_mse_mean",
    "mean_derived_frame_symmetric_rmse",
    "derived_rmse_from_mean_symmetric_mse",
    "derived_rmse_aggregation_gap",
    "mean_F_5",
    "mean_F_10",
    "mean_F_20",
    "mean_F_30",
    "mean_cd_rmse_symmetric",
    "mean_pred_to_gt_rmse",
    "mean_gt_to_pred_rmse",
    "mean_cd_rmse_symmetric_norm_bbox",
    "mean_pred_to_gt_rmse_norm_bbox",
    "mean_gt_to_pred_rmse_norm_bbox",
    "mean_point_count_ratio",
    "mean_psnr_y",
    "mean_psnr_u",
    "mean_psnr_v",
    "mean_pcqm",
    "mean_projection_ssim_6view",
    "mean_projection_lpips_6view",
    "runtime_seconds_sum",
    "runtime_seconds_mean",
)


def load_json(path: Path | None) -> dict:
    if path is None or not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def collect_runtime(runtime_glob: str | None) -> dict:
    if not runtime_glob:
        return {}
    stage_stats = {}
    total_records = 0
    total_seconds = 0.0
    end_to_end_mean = 0.0
    for path_text in sorted(glob.glob(runtime_glob)):
        path = Path(path_text)
        records_by_frame = {}
        anonymous_seconds = []
        for record in read_jsonl(path):
            value = record.get("seconds") if isinstance(record, dict) else None
            if isinstance(value, (int, float)) and math.isfinite(float(value)):
                frame_key = next(
                    (
                        str(record[key])
                        for key in ("output", "frame", "input", "source", "base")
                        if key in record
                    ),
                    None,
                )
                if frame_key is None:
                    anonymous_seconds.append(float(value))
                else:
                    # Runtime logs are append-only; keep the latest rerun of a frame.
                    records_by_frame[frame_key] = float(value)
        seconds = list(records_by_frame.values()) + anonymous_seconds
        if not seconds:
            continue
        stage_sum = float(sum(seconds))
        stage_mean = stage_sum / len(seconds)
        stage_stats[path.stem] = {
            "records": len(seconds),
            "seconds_sum": stage_sum,
            "seconds_mean": stage_mean,
        }
        total_records += len(seconds)
        total_seconds += stage_sum
        end_to_end_mean += stage_mean
    if not stage_stats:
        return {}
    return {
        "runtime_record_count": total_records,
        "runtime_stage_count": len(stage_stats),
        "runtime_seconds_sum": total_seconds,
        "runtime_seconds_mean": end_to_end_mean,
        "runtime_stage_stats": stage_stats,
        "runtime_definition": "End-to-end mean is the sum of per-stage mean frame times after deduplicating rerun records.",
    }


def finite_values(record: dict, keys: tuple[str, ...]) -> list[float]:
    values = []
    for key in keys:
        value = record.get(key)
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            values.append(float(value))
    return values


def add_group_means(board: dict) -> None:
    geometry = finite_values(
        board,
        (
            "mean_CD_Acc",
            "mean_CD_Comp",
            "mean_cd_rmse_symmetric_norm_bbox",
            "mean_pred_to_gt_rmse_norm_bbox",
            "mean_gt_to_pred_rmse_norm_bbox",
        ),
    )
    texture = finite_values(board, ("mean_psnr_y", "mean_psnr_u", "mean_psnr_v"))
    perceptual = finite_values(board, ("mean_pcqm", "mean_projection_ssim_6view"))
    if geometry:
        board["geometry_norm_rmse_proxy_mean"] = float(sum(geometry) / len(geometry))
    if texture:
        board["texture_psnr_proxy_mean"] = float(sum(texture) / len(texture))
    if perceptual:
        board["perceptual_proxy_mean"] = float(sum(perceptual) / len(perceptual))


def write_markdown(path: Path, board: dict) -> None:
    lines = [f"# Validation Board: {board['run_name']}", ""]
    lines.append("| Metric | Value |")
    lines.append("|---|---:|")
    for key in BOARD_KEYS + (
        "geometry_norm_rmse_proxy_mean",
        "texture_psnr_proxy_mean",
        "perceptual_proxy_mean",
    ):
        if key in board:
            value = board[key]
            if isinstance(value, float):
                value_text = f"{value:.6g}"
            else:
                value_text = str(value)
            lines.append(f"| `{key}` | {value_text} |")
    lines.append("")
    lines.append("Lower is better for RMSE and LPIPS; higher is better for F-score, PSNR, PCQM, and SSIM.")
    lines.append("Geometry keys prefer UVG-CWI Metric output when `uvg_metric_count` is present; PCQM is supplementary.")
    lines.append("`chamfer-L2` is the official sum of directional MSEs; symmetric MSE/RMSE fields are diagnostics, not extra official metrics.")
    lines.append("No official normalization or final weighted score is inferred here.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize DQPC validation metrics")
    parser.add_argument("--run-name", default="dqpc_run")
    parser.add_argument("--basic-summary", default=None, type=Path)
    parser.add_argument("--uvg-metric-summary", default=None, type=Path)
    parser.add_argument("--external-summary", default=None, type=Path)
    parser.add_argument("--projection-summary", default=None, type=Path)
    parser.add_argument("--runtime-glob", default=None)
    parser.add_argument("--out-json", required=True, type=Path)
    parser.add_argument("--out-md", default=None, type=Path)
    args = parser.parse_args()

    board = {"run_name": args.run_name}
    for source in (
        load_json(args.basic_summary),
        load_json(args.uvg_metric_summary),
        load_json(args.external_summary),
        load_json(args.projection_summary),
        collect_runtime(args.runtime_glob),
    ):
        board.update(source)

    add_group_means(board)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(board, indent=2), encoding="utf-8")
    if args.out_md:
        write_markdown(args.out_md, board)
    print(json.dumps(board, indent=2))


if __name__ == "__main__":
    main()
