#!/usr/bin/env python3
"""Run PU-Dense geometry inference on DQPC CG frames."""

from __future__ import annotations

import argparse
import glob
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree

from dqpc_data import (
    dequantize_xyz,
    infer_voxel_size_from_paths,
    kdtree_partition_indices,
    quantize_xyz,
    read_ply_xyz,
    unique_int_coords,
    write_ply_xyz,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DQPC PU-Dense geometry inference")
    parser.add_argument("--pudense-root", default="external/PointCloudUpsampling", type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--cg-glob", required=True)
    parser.add_argument("--input-root", required=True, type=Path, help="Root used to preserve relative output layout")
    parser.add_argument("--geometry-out-root", required=True, type=Path)
    parser.add_argument("--runtime-log", default=None, type=Path)
    parser.add_argument(
        "--voxel-size",
        default="auto",
        help="Quantization voxel size. Use 'auto' for coordinate-unit detection.",
    )
    parser.add_argument("--max-points-per-block", default=70000, type=int)
    parser.add_argument("--up-ratio", default=4.0, type=float)
    parser.add_argument(
        "--target-point-ratio",
        default=None,
        type=float,
        help="Override top-k output count per block as input_points * ratio. Defaults to --up-ratio.",
    )
    parser.add_argument(
        "--score-threshold",
        default=None,
        type=float,
        help="Optional sigmoid threshold on PU-Dense logits. If set, thresholded points replace top-k pruning.",
    )
    parser.add_argument(
        "--max-output-point-ratio",
        default=0.0,
        type=float,
        help="Optional cap after thresholding, as input_points * ratio. <=0 disables.",
    )
    parser.add_argument(
        "--block-halo",
        default=32,
        type=int,
        help="Extra quantized voxels of context around each KD-tree block before inference.",
    )
    parser.add_argument(
        "--include-input",
        action="store_true",
        help="Union quantized CG input points with PU-Dense output before dequantization.",
    )
    parser.add_argument("--last-kernel-size", default=5, type=int)
    parser.add_argument("--limit", default=0, type=int)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--ascii", action="store_true")
    parser.add_argument("--seed", default=1, type=int)
    return parser.parse_args()


def halo_block(
    all_coords: np.ndarray,
    block: np.ndarray,
    halo: int,
    spatial_tree: cKDTree | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lo = block.min(axis=0)
    hi = block.max(axis=0)
    if halo <= 0:
        return block, lo, hi
    halo_lo = lo - halo
    halo_hi = hi + halo
    if spatial_tree is None:
        candidates = all_coords
    else:
        center = 0.5 * (halo_lo + halo_hi)
        radius = float(np.nextafter(np.linalg.norm(0.5 * (halo_hi - halo_lo)), np.inf))
        candidate_idx = spatial_tree.query_ball_point(center, radius, workers=-1)
        candidates = all_coords[np.asarray(candidate_idx, dtype=np.int64)]
    mask = np.all((candidates >= halo_lo[None, :]) & (candidates <= halo_hi[None, :]), axis=1)
    return candidates[mask], lo, hi


def filter_to_bbox(coords: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    if coords.shape[0] == 0:
        return coords
    mask = np.all((coords >= lo[None, :]) & (coords <= hi[None, :]), axis=1)
    return coords[mask]


def relative_output_path(input_root: Path, frame_path: Path, out_root: Path) -> Path:
    try:
        rel = frame_path.resolve().relative_to(input_root.resolve())
    except ValueError:
        rel = Path(frame_path.name)
    return out_root / rel


def main() -> None:
    args = parse_args()
    sys.path.insert(0, str(args.pudense_root.resolve()))

    import MinkowskiEngine as ME
    from model.Network import MyNet

    frame_paths = sorted(Path(p) for p in glob.glob(args.cg_glob))
    if args.limit > 0:
        frame_paths = frame_paths[: args.limit]
    if not frame_paths:
        raise RuntimeError(f"No CG frames found from {args.cg_glob}")
    voxel_size = infer_voxel_size_from_paths(frame_paths, args.voxel_size)
    print({"resolved_voxel_size": voxel_size, "voxel_size_arg": str(args.voxel_size), "frame_count": len(frame_paths)})

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MyNet(last_kernel_size=args.last_kernel_size).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    if args.runtime_log:
        args.runtime_log.parent.mkdir(parents=True, exist_ok=True)

    for frame_idx, frame_path in enumerate(frame_paths, start=1):
        tic = time.time()
        out_path = relative_output_path(args.input_root, frame_path, args.geometry_out_root)
        if args.skip_existing and out_path.exists():
            print(f"[{frame_idx}/{len(frame_paths)}] skip existing {out_path}")
            continue

        xyz = read_ply_xyz(frame_path)
        coords, origin = quantize_xyz(xyz, voxel_size)
        coords = unique_int_coords(coords)
        parts = kdtree_partition_indices(coords, args.max_points_per_block)
        spatial_tree = cKDTree(coords) if args.block_halo > 0 else None

        outputs: list[np.ndarray] = []
        target_ratio = args.target_point_ratio if args.target_point_ratio is not None else args.up_ratio
        for part_idx, idx in enumerate(parts, start=1):
            block = coords[idx]
            block_input, keep_lo, keep_hi = halo_block(coords, block, args.block_halo, spatial_tree)
            p = ME.utils.batched_coordinates([block_input])
            feats = torch.ones((p.shape[0], 1), dtype=torch.float32)
            target_count = max(1, round(block.shape[0] * target_ratio))
            x = ME.SparseTensor(feats=feats, coords=p).to(device)
            with torch.no_grad():
                if args.score_threshold is None:
                    out, _, _, _ = model(
                        x,
                        coords_T=None,
                        device=device,
                        prune=True,
                        target_counts=[target_count],
                    )
                    block_out = out.C[:, 1:].detach().cpu().numpy().astype(np.int32)
                else:
                    _, out_cls, _, _ = model(x, coords_T=None, device=device, prune=False)
                    score = torch.sigmoid(out_cls.F.view(-1))
                    keep = score >= args.score_threshold
                    if args.max_output_point_ratio > 0 and int(keep.sum()) > round(block.shape[0] * args.max_output_point_ratio):
                        cap = max(1, round(block.shape[0] * args.max_output_point_ratio))
                        _, top_idx = torch.topk(score, cap)
                        keep = torch.zeros_like(score, dtype=torch.bool)
                        keep[top_idx] = True
                    block_out = out_cls.C[keep, 1:].detach().cpu().numpy().astype(np.int32)
            block_out = filter_to_bbox(block_out, keep_lo, keep_hi)
            outputs.append(block_out)
            print(
                f"[{frame_idx}/{len(frame_paths)}] {frame_path.name} block {part_idx}/{len(parts)} "
                f"input={block.shape[0]} halo_input={block_input.shape[0]} output={block_out.shape[0]}"
            )

        if args.include_input:
            outputs.append(coords)
        rec_coords = unique_int_coords(np.concatenate(outputs, axis=0))
        rec_xyz = dequantize_xyz(rec_coords, voxel_size, origin)
        write_ply_xyz(out_path, rec_xyz, text=args.ascii)

        record = {
            "frame": str(frame_path),
            "output": str(out_path),
            "input_points": int(xyz.shape[0]),
            "quantized_input_points": int(coords.shape[0]),
            "output_points": int(rec_xyz.shape[0]),
            "point_count_ratio_vs_input": float(rec_xyz.shape[0] / max(coords.shape[0], 1)),
            "include_input": bool(args.include_input),
            "blocks": len(parts),
            "block_halo": int(args.block_halo),
            "score_threshold": args.score_threshold,
            "seconds": round(time.time() - tic, 4),
        }
        print(record)
        if args.runtime_log:
            with args.runtime_log.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")


if __name__ == "__main__":
    main()
