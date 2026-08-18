#!/usr/bin/env python3
"""Six-view projection SSIM and optional LPIPS for DQPC outputs.

This is a local proxy for the projection-based perceptual metrics described by
the challenge PDF. It is not the official scoring code.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from dqpc_data import matching_he_path, read_ply_xyz_rgb


def relative_path(root: Path, path: Path) -> Path:
    try:
        return path.resolve().relative_to(root.resolve())
    except ValueError:
        return Path(path.name)


VIEW_SPECS = {
    "front": (0, 1, 2, 1.0),
    "back": (0, 1, 2, -1.0),
    "right": (2, 1, 0, 1.0),
    "left": (2, 1, 0, -1.0),
    "top": (0, 2, 1, 1.0),
    "bottom": (0, 2, 1, -1.0),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Projection-based DQPC evaluation")
    parser.add_argument("--pred-root", required=True, type=Path)
    parser.add_argument("--pred-glob", required=True)
    parser.add_argument("--gt-root", required=True, type=Path)
    parser.add_argument("--out-jsonl", required=True, type=Path)
    parser.add_argument("--summary-json", default=None, type=Path)
    parser.add_argument("--image-size", default=512, type=int)
    parser.add_argument("--lpips", action="store_true", help="Compute LPIPS if the lpips package is installed")
    parser.add_argument("--limit", default=0, type=int)
    return parser.parse_args()


def global_ssim(img_a: np.ndarray, img_b: np.ndarray) -> float:
    a = img_a.astype(np.float64)
    b = img_b.astype(np.float64)
    c1 = (0.01 * 255.0) ** 2
    c2 = (0.03 * 255.0) ** 2
    vals = []
    for channel in range(3):
        x = a[:, :, channel]
        y = b[:, :, channel]
        ux = float(x.mean())
        uy = float(y.mean())
        vx = float(x.var())
        vy = float(y.var())
        cov = float(((x - ux) * (y - uy)).mean())
        denom = (ux * ux + uy * uy + c1) * (vx + vy + c2)
        vals.append(((2.0 * ux * uy + c1) * (2.0 * cov + c2)) / denom if denom > 0 else 1.0)
    return float(np.mean(vals))


def ssim_image(img_a: np.ndarray, img_b: np.ndarray) -> tuple[float, str]:
    try:
        from skimage.metrics import structural_similarity

        return float(structural_similarity(img_a, img_b, channel_axis=-1, data_range=255)), "skimage"
    except Exception:
        return global_ssim(img_a, img_b), "global"


def render_view(
    xyz: np.ndarray,
    rgb: np.ndarray,
    bounds_min: np.ndarray,
    bounds_max: np.ndarray,
    image_size: int,
    view_name: str,
) -> np.ndarray:
    axis_x, axis_y, axis_depth, depth_sign = VIEW_SPECS[view_name]
    span = np.maximum(bounds_max - bounds_min, 1e-12)
    px = np.floor((xyz[:, axis_x] - bounds_min[axis_x]) / span[axis_x] * (image_size - 1)).astype(np.int64)
    py = np.floor((xyz[:, axis_y] - bounds_min[axis_y]) / span[axis_y] * (image_size - 1)).astype(np.int64)
    px = np.clip(px, 0, image_size - 1)
    py = np.clip(image_size - 1 - py, 0, image_size - 1)
    flat = py * image_size + px
    depth = xyz[:, axis_depth] * depth_sign

    order = np.lexsort((-depth, flat))
    flat_sorted = flat[order]
    keep = np.ones(order.shape[0], dtype=bool)
    keep[1:] = flat_sorted[1:] != flat_sorted[:-1]
    selected = order[keep]

    img = np.zeros((image_size * image_size, 3), dtype=np.uint8)
    img[flat[selected]] = np.clip(np.rint(rgb[selected]), 0, 255).astype(np.uint8)
    return img.reshape(image_size, image_size, 3)


def make_lpips_model() -> Any | None:
    try:
        import lpips
        import torch

        model = lpips.LPIPS(net="alex")
        model.eval()
        return model
    except Exception:
        return None


def lpips_distance(model: Any, img_a: np.ndarray, img_b: np.ndarray) -> float:
    import torch

    a = torch.from_numpy(img_a.astype(np.float32) / 127.5 - 1.0).permute(2, 0, 1).unsqueeze(0)
    b = torch.from_numpy(img_b.astype(np.float32) / 127.5 - 1.0).permute(2, 0, 1).unsqueeze(0)
    with torch.no_grad():
        return float(model(a, b).item())


def find_gt(pred_root: Path, pred: Path, gt_root: Path) -> Path:
    rel = relative_path(pred_root, pred)
    gt = matching_he_path(gt_root, rel) or (gt_root / rel)
    if not gt.exists():
        raise FileNotFoundError(f"Missing GT for {pred}: {gt}")
    return gt


def eval_pair(pred_path: Path, gt_path: Path, image_size: int, lpips_model: Any | None) -> dict:
    pred_xyz, pred_rgb = read_ply_xyz_rgb(pred_path)
    gt_xyz, gt_rgb = read_ply_xyz_rgb(gt_path)
    if pred_rgb is None or gt_rgb is None:
        raise ValueError(f"Projection metrics require RGB in both files: {pred_path}, {gt_path}")

    merged = np.vstack([pred_xyz, gt_xyz])
    bounds_min = merged.min(axis=0)
    bounds_max = merged.max(axis=0)

    views: dict[str, dict[str, float]] = {}
    ssim_values = []
    lpips_values = []
    ssim_method = "global"
    for view_name in VIEW_SPECS:
        pred_img = render_view(pred_xyz, pred_rgb, bounds_min, bounds_max, image_size, view_name)
        gt_img = render_view(gt_xyz, gt_rgb, bounds_min, bounds_max, image_size, view_name)
        ssim_value, method = ssim_image(pred_img, gt_img)
        ssim_method = method
        view_record = {"ssim": ssim_value}
        ssim_values.append(ssim_value)
        if lpips_model is not None:
            lpips_value = lpips_distance(lpips_model, pred_img, gt_img)
            view_record["lpips"] = lpips_value
            lpips_values.append(lpips_value)
        views[view_name] = view_record

    record = {
        "pred": str(pred_path),
        "gt": str(gt_path),
        "image_size": image_size,
        "ssim_method": ssim_method,
        "projection_ssim_6view": float(np.mean(ssim_values)),
        "projection_views": views,
    }
    if lpips_values:
        record["projection_lpips_6view"] = float(np.mean(lpips_values))
    return record


def summarize(records: list[dict]) -> dict:
    summary = {"count": len(records)}
    for key in ("projection_ssim_6view", "projection_lpips_6view"):
        vals = [float(r[key]) for r in records if key in r and math.isfinite(float(r[key]))]
        if vals:
            summary[f"mean_{key}"] = float(np.mean(vals))
    for view_name in VIEW_SPECS:
        vals = [float(r["projection_views"][view_name]["ssim"]) for r in records]
        if vals:
            summary[f"mean_projection_ssim_{view_name}"] = float(np.mean(vals))
        lpips_vals = [
            float(r["projection_views"][view_name]["lpips"])
            for r in records
            if "lpips" in r["projection_views"][view_name]
        ]
        if lpips_vals:
            summary[f"mean_projection_lpips_{view_name}"] = float(np.mean(lpips_vals))
    return summary


def main() -> None:
    args = parse_args()
    preds = sorted(Path(p) for p in glob.glob(args.pred_glob))
    if args.limit > 0:
        preds = preds[: args.limit]
    if not preds:
        raise RuntimeError(f"No prediction files found from {args.pred_glob}")

    lpips_model = make_lpips_model() if args.lpips else None
    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)

    records = []
    with args.out_jsonl.open("w", encoding="utf-8") as f:
        for pred in preds:
            gt = find_gt(args.pred_root, pred, args.gt_root)
            record = eval_pair(pred, gt, args.image_size, lpips_model)
            if args.lpips and lpips_model is None:
                record["lpips_status"] = "skipped: install the lpips package and its torch weights"
            records.append(record)
            f.write(json.dumps(record) + "\n")
            print(json.dumps({"pred": str(pred), "projection_ssim_6view": record["projection_ssim_6view"]}))

    summary = summarize(records)
    print(json.dumps(summary, indent=2))
    if args.summary_json:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
