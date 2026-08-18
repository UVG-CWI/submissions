#!/usr/bin/env python3
"""Run the bundled UVG-CWI Metric implementation on DQPC prediction PLYs."""

from __future__ import annotations

import argparse
import glob
import hashlib
import importlib.util
import json
import math
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np

from dqpc_data import distance_scale_from_paths


CG_TO_HE_TOKENS = (
    ("/CG/15fps/", "/HE/15fps/"),
    ("/CGv2/15fps/", "/HE/15fps/"),
    ("/CGv2_15/", "/HE/15fps/"),
    ("/CG_aligned/15fps/", "/HE/15fps/"),
    ("/consumer-grade_capture_system/CG/15fps/", "/high-end_capture_system/HE/15fps/"),
    ("/consumer-grade_capture_system/CGv2/15fps/", "/high-end_capture_system/HE/15fps/"),
    ("/consumer-grade_capture_system/CG_aligned/15fps/", "/high-end_capture_system/HE/15fps/"),
)


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


def relative_path(root: Path, path: Path) -> Path:
    try:
        return path.resolve().relative_to(root.resolve())
    except ValueError:
        return Path(path.name)


def parse_thresholds(text: str) -> list[int | float]:
    thresholds: list[int | float] = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        value = float(item)
        thresholds.append(int(value) if value.is_integer() else value)
    if not thresholds:
        raise ValueError("At least one F-score threshold is required")
    return sorted(set(thresholds))


def canonical_threshold(value: float) -> int | float:
    return int(value) if float(value).is_integer() else float(value)


def remap_threshold_metrics(
    metrics: dict[str, Any],
    requested: list[int | float],
    resolved: list[int | float],
) -> dict[str, Any]:
    output = dict(metrics)
    for requested_value, resolved_value in zip(requested, resolved):
        for prefix in ("P", "R", "F"):
            resolved_key = f"{prefix}_{resolved_value}"
            requested_key = f"{prefix}_{requested_value}"
            if resolved_key in output:
                output[requested_key] = output.pop(resolved_key)
    return output


def load_uvg_metric(metric_root: Path) -> ModuleType:
    metric_file = metric_root / "metrics.py"
    if not metric_file.exists():
        raise FileNotFoundError(f"Missing UVG-CWI Metric file: {metric_file}")
    spec = importlib.util.spec_from_file_location("uvg_cwi_metric_metrics", metric_file)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load UVG-CWI Metric module from {metric_file}")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except ModuleNotFoundError as exc:
        missing = exc.name or str(exc)
        raise RuntimeError(
            "UVG-CWI Metric dependency is missing. Use a PYTHON_BIN environment "
            f"with scipy and trimesh installed; first missing module: {missing}"
        ) from exc
    return module


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_blob_sha1(path: Path) -> str:
    content = path.read_bytes()
    header = f"blob {len(content)}\0".encode("ascii")
    return hashlib.sha1(header + content).hexdigest()


def json_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (int, str)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(k): json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(v) for v in value]
    return str(value)


def finite_numeric_items(record: dict[str, Any]) -> dict[str, float]:
    items: dict[str, float] = {}
    for key, value in record.items():
        if isinstance(value, np.generic):
            value = value.item()
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            items[key] = float(value)
    return items


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate DQPC outputs with external/UVG-CWI-Metric")
    parser.add_argument("--pred-root", required=True, type=Path)
    parser.add_argument("--pred-glob", required=True)
    parser.add_argument("--gt-root", required=True, type=Path)
    parser.add_argument("--metric-root", required=True, type=Path)
    parser.add_argument("--out-jsonl", required=True, type=Path)
    parser.add_argument("--summary-json", default=None, type=Path)
    parser.add_argument(
        "--fscore-thresholds",
        default="5,10,20,30",
        help="Comma-separated distance thresholds passed to UVG-CWI Metric.",
    )
    parser.add_argument("--limit", default=0, type=int)
    parser.add_argument("--threshold-unit", choices=("mm", "coordinate"), default="mm")
    args = parser.parse_args()

    metric_module = load_uvg_metric(args.metric_root)
    metric_file = args.metric_root / "metrics.py"
    metric_sha256 = file_sha256(metric_file)
    metric_git_blob = git_blob_sha1(metric_file)
    requested_thresholds = parse_thresholds(args.fscore_thresholds)

    preds = sorted(Path(p) for p in glob.glob(args.pred_glob))
    if args.limit > 0:
        preds = preds[: args.limit]
    if not preds:
        raise RuntimeError(f"No prediction files found from {args.pred_glob}")
    threshold_scale = distance_scale_from_paths(preds, args.threshold_unit)
    resolved_thresholds = [
        canonical_threshold(float(value) * threshold_scale)
        for value in requested_thresholds
    ]

    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    numeric_by_key: dict[str, list[float]] = {}
    with args.out_jsonl.open("w", encoding="utf-8") as f:
        for pred in preds:
            rel = relative_path(args.pred_root, pred)
            gt = matching_he_path(args.gt_root, rel) or (args.gt_root / rel)
            if not gt.exists():
                raise FileNotFoundError(f"Missing GT for {pred}: {gt}")

            raw_metrics = metric_module.eval_pointcloud(
                pre_mesh_ply=str(pred),
                gt_mesh_ply=str(gt),
                samplepoint=0,
                eval_type="ply",
                thresholds=resolved_thresholds,
            )
            metrics = remap_threshold_metrics(
                raw_metrics,
                requested_thresholds,
                resolved_thresholds,
            )
            chamfer_l2 = metrics.get("chamfer-L2")
            derived = {}
            if isinstance(chamfer_l2, (int, float, np.generic)):
                symmetric_mse = 0.5 * float(chamfer_l2)
                derived = {
                    "derived_symmetric_mse_mean": symmetric_mse,
                    "derived_frame_symmetric_rmse": math.sqrt(max(symmetric_mse, 0.0)),
                }
            record = {
                "pred": str(pred),
                "gt": str(gt),
                "metric_source": str(metric_file.resolve()),
                "metric_source_sha256": metric_sha256,
                "metric_source_git_blob_sha1": metric_git_blob,
                **metrics,
                **derived,
            }
            for key, value in finite_numeric_items({**metrics, **derived}).items():
                numeric_by_key.setdefault(key, []).append(value)
            records.append(record)
            safe_record = json_value(record)
            f.write(json.dumps(safe_record) + "\n")
            print(json.dumps(safe_record))

    summary: dict[str, Any] = {
        "uvg_metric_count": len(records),
        "uvg_metric_source": str(metric_file.resolve()),
        "uvg_metric_source_sha256": metric_sha256,
        "uvg_metric_source_git_blob_sha1": metric_git_blob,
        "uvg_metric_thresholds": requested_thresholds,
        "uvg_metric_threshold_unit": args.threshold_unit,
        "uvg_metric_resolved_thresholds": resolved_thresholds,
        "uvg_metric_definitions": {
            "mean_chamfer-L2": "Official value: sum of the two directional mean squared nearest-neighbor distances.",
            "mean_derived_symmetric_mse_mean": "Diagnostic only: 0.5 * official chamfer-L2.",
            "mean_derived_frame_symmetric_rmse": "Diagnostic only: average of per-frame sqrt(0.5 * chamfer-L2).",
        },
    }
    for key, values in sorted(numeric_by_key.items()):
        if values:
            summary[f"mean_{key}"] = float(np.mean(values))
    mean_symmetric_mse = summary.get("mean_derived_symmetric_mse_mean")
    mean_frame_rmse = summary.get("mean_derived_frame_symmetric_rmse")
    if isinstance(mean_symmetric_mse, (int, float)):
        summary["derived_rmse_from_mean_symmetric_mse"] = math.sqrt(max(mean_symmetric_mse, 0.0))
    if isinstance(mean_frame_rmse, (int, float)) and "derived_rmse_from_mean_symmetric_mse" in summary:
        summary["derived_rmse_aggregation_gap"] = (
            summary["derived_rmse_from_mean_symmetric_mse"] - mean_frame_rmse
        )

    safe_summary = json_value(summary)
    print(json.dumps(safe_summary, indent=2))
    if args.summary_json:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(json.dumps(safe_summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
