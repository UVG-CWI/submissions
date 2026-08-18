#!/usr/bin/env python3
"""Run GQE-Net color refinement on DQPC enhanced colored PLY files."""

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

from dqpc_data import read_ply_xyz_rgb, write_ply_xyz_rgb
from gqenet_dqpc import patch_indices, relative_path, rgb_to_yuv, yuv_to_rgb


CHANNEL_NAMES = ("y", "u", "v")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Infer GQE-Net DQPC color refinement")
    parser.add_argument("--gqenet-root", default="external/GQE-Net", type=Path)
    parser.add_argument("--input-root", required=True, type=Path)
    parser.add_argument("--input-glob", required=True)
    parser.add_argument("--out-root", required=True, type=Path)
    parser.add_argument("--model-y", required=True, type=Path)
    parser.add_argument("--model-u", required=True, type=Path)
    parser.add_argument("--model-v", required=True, type=Path)
    parser.add_argument("--patch-size", default=2048, type=int)
    parser.add_argument("--patch-stride", default=1024, type=int)
    parser.add_argument("--batch-size", default=8, type=int)
    parser.add_argument(
        "--center-mode",
        choices=("v3_stride", "spatial_voxel"),
        default="v3_stride",
        help="v3_stride preserves V3 point-order sampling; spatial_voxel is an opt-in coverage experiment.",
    )
    parser.add_argument("--coord-scale", default=1.0, type=float)
    parser.add_argument(
        "--prediction-mode",
        choices=("auto", "absolute", "residual"),
        default="auto",
        help="Interpret checkpoint output. Auto reads checkpoint metadata and treats legacy checkpoints as absolute.",
    )
    parser.add_argument(
        "--residual",
        action="store_true",
        help="Deprecated compatibility alias for --prediction-mode residual.",
    )
    parser.add_argument("--blend-y", default=1.0, type=float, help="Blend strength from input Y to predicted Y.")
    parser.add_argument("--blend-u", default=1.0, type=float, help="Blend strength from input U to predicted U.")
    parser.add_argument("--blend-v", default=1.0, type=float, help="Blend strength from input V to predicted V.")
    parser.add_argument("--max-delta-y", default=0.0, type=float, help="Optional absolute Y correction cap; <=0 disables.")
    parser.add_argument("--max-delta-u", default=0.0, type=float, help="Optional absolute U correction cap; <=0 disables.")
    parser.add_argument("--max-delta-v", default=0.0, type=float, help="Optional absolute V correction cap; <=0 disables.")
    parser.add_argument("--limit", default=0, type=int)
    parser.add_argument("--runtime-log", default=None, type=Path)
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def checkpoint_prediction_mode(ckpt: dict) -> str:
    mode = ckpt.get("prediction_mode")
    if mode in {"absolute", "residual"}:
        return str(mode)
    if "residual" in ckpt:
        return "residual" if bool(ckpt["residual"]) else "absolute"
    return "absolute"


def resolve_prediction_mode(ckpt: dict, requested_mode: str) -> str:
    if requested_mode in {"absolute", "residual"}:
        return requested_mode
    return checkpoint_prediction_mode(ckpt)


def validate_checkpoint_config(ckpt: dict, path: Path, channel: int, patch_size: int, coord_scale: float) -> None:
    if "channel" in ckpt and int(ckpt["channel"]) != channel:
        raise ValueError(
            f"{path} is a {CHANNEL_NAMES[int(ckpt['channel'])]}-channel checkpoint, "
            f"but it was passed as the {CHANNEL_NAMES[channel]} model"
        )
    if "patch_size" in ckpt and int(ckpt["patch_size"]) != patch_size:
        raise ValueError(
            f"{path} was trained with patch_size={ckpt['patch_size']}, "
            f"but inference uses patch_size={patch_size}"
        )
    if "coord_scale" in ckpt and not np.isclose(float(ckpt["coord_scale"]), coord_scale):
        raise ValueError(
            f"{path} was trained with coord_scale={ckpt['coord_scale']}, "
            f"but inference uses coord_scale={coord_scale}"
        )


def load_model(
    model_cls,
    path: Path,
    device: torch.device,
    channel: int,
    requested_mode: str,
    patch_size: int,
    coord_scale: float,
):
    model = model_cls().to(device)
    ckpt = torch.load(path, map_location=device)
    validate_checkpoint_config(ckpt, path, channel, patch_size, coord_scale)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, resolve_prediction_mode(ckpt, requested_mode)


def select_centers(xyz: np.ndarray, patch_size: int, patch_stride: int, mode: str) -> np.ndarray:
    if xyz.shape[0] <= patch_size:
        return xyz[:1]
    if mode == "v3_stride":
        return xyz[:: max(1, patch_stride)]

    target_count = max(1, int(np.ceil(xyz.shape[0] / max(1, patch_stride))))
    lo = xyz.min(axis=0)
    span = np.maximum(xyz.max(axis=0) - lo, 1e-12)
    resolution = max(1, int(np.ceil(target_count ** (1.0 / 3.0))))
    cell = np.floor((xyz - lo) / span * resolution).astype(np.int64)
    cell = np.clip(cell, 0, resolution - 1)
    keys = cell[:, 0] + resolution * (cell[:, 1] + resolution * cell[:, 2])
    _, representative = np.unique(keys, return_index=True)
    representative.sort()
    if representative.shape[0] > target_count:
        keep = np.linspace(0, representative.shape[0] - 1, target_count, dtype=np.int64)
        representative = representative[keep]
    elif representative.shape[0] < target_count:
        fallback = np.arange(0, xyz.shape[0], max(1, patch_stride), dtype=np.int64)
        representative = np.unique(np.concatenate([representative, fallback]))[:target_count]
    return xyz[representative]


def run_models(
    models,
    xyz: np.ndarray,
    yuv: np.ndarray,
    centers: np.ndarray,
    args,
    device,
) -> tuple[np.ndarray, np.ndarray]:
    tree = cKDTree(xyz)
    accum = np.zeros((3, xyz.shape[0]), dtype=np.float64)
    counts = np.zeros((xyz.shape[0],), dtype=np.float64)
    for start in range(0, centers.shape[0], max(1, args.batch_size)):
        center_batch = centers[start : start + max(1, args.batch_size)]
        if xyz.shape[0] < args.patch_size:
            batch_indices = np.stack(
                [patch_indices(tree, xyz, center, args.patch_size) for center in center_batch],
                axis=0,
            )
        else:
            _, batch_indices = tree.query(center_batch, k=args.patch_size, workers=-1)
            batch_indices = np.asarray(batch_indices, dtype=np.int64).reshape(center_batch.shape[0], -1)
        patch_xyz = xyz[batch_indices]
        patch_center = patch_xyz.mean(axis=1, keepdims=True)
        patch_xyz_norm = ((patch_xyz - patch_center) / max(float(args.coord_scale), 1e-12)).astype(np.float32)
        predictions = []
        with torch.no_grad():
            for channel, model in enumerate(models):
                data_np = np.concatenate(
                    [patch_xyz_norm, yuv[batch_indices, channel][..., None]],
                    axis=2,
                ).astype(np.float32, copy=False)
                data = torch.from_numpy(data_np).to(device).permute(0, 2, 1).contiguous()
                predictions.append(model(data).detach().cpu().numpy()[:, :, 0])
        for batch_row, patch_idx in enumerate(batch_indices):
            counts[patch_idx] += 1.0
            for channel in range(3):
                accum[channel, patch_idx] += predictions[channel][batch_row]

    covered = counts > 0
    out = np.zeros((xyz.shape[0], 3), dtype=np.float32)
    out[covered] = (accum[:, covered] / counts[covered]).T.astype(np.float32)
    return out, covered


def compose_channel_prediction(
    input_channel: np.ndarray,
    model_output: np.ndarray,
    covered: np.ndarray,
    prediction_mode: str,
    blend: float,
    max_delta: float,
) -> np.ndarray:
    if not 0.0 <= blend <= 1.0:
        raise ValueError(f"Blend must be in [0, 1], got {blend}")
    if prediction_mode not in {"absolute", "residual"}:
        raise ValueError(f"Unsupported prediction mode: {prediction_mode}")

    output = input_channel.astype(np.float32).copy()
    if not np.any(covered):
        return output
    if prediction_mode == "residual":
        candidate = input_channel[covered] + model_output[covered]
    else:
        candidate = model_output[covered]
    delta = candidate - input_channel[covered]
    if max_delta > 0:
        delta = np.clip(delta, -max_delta, max_delta)
    output[covered] = input_channel[covered] + blend * delta
    return np.clip(output, 0.0, 255.0)


def main() -> None:
    args = parse_args()
    if args.residual:
        if args.prediction_mode == "absolute":
            raise ValueError("--residual conflicts with --prediction-mode absolute")
        args.prediction_mode = "residual"

    sys.path.insert(0, str(args.gqenet_root.resolve()))
    import model_GQE_Net_final as gqe_model

    from model_GQE_Net_final import GAPCN

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gqe_model.devices = str(device)
    model_paths = (args.model_y, args.model_u, args.model_v)
    loaded = [
        load_model(
            GAPCN,
            path,
            device,
            channel,
            args.prediction_mode,
            args.patch_size,
            args.coord_scale,
        )
        for channel, path in enumerate(model_paths)
    ]
    models = [item[0] for item in loaded]
    prediction_modes = [item[1] for item in loaded]
    blends = (args.blend_y, args.blend_u, args.blend_v)
    max_deltas = (args.max_delta_y, args.max_delta_u, args.max_delta_v)
    print(
        {
            "device": str(device),
            "prediction_modes": dict(zip(CHANNEL_NAMES, prediction_modes)),
            "blends": dict(zip(CHANNEL_NAMES, blends)),
            "max_deltas": dict(zip(CHANNEL_NAMES, max_deltas)),
        }
    )
    frames = sorted(Path(p) for p in glob.glob(args.input_glob))
    if args.limit > 0:
        frames = frames[: args.limit]
    if not frames:
        raise RuntimeError(f"No input PLY files found from {args.input_glob}")
    if args.runtime_log:
        args.runtime_log.parent.mkdir(parents=True, exist_ok=True)

    for frame_idx, frame_path in enumerate(frames, start=1):
        tic = time.time()
        out_path = args.out_root / relative_path(args.input_root, frame_path)
        if args.skip_existing and out_path.exists():
            print(f"[{frame_idx}/{len(frames)}] skip existing {out_path}")
            continue
        xyz, rgb = read_ply_xyz_rgb(frame_path)
        if rgb is None:
            raise ValueError(f"{frame_path} is missing RGB")
        yuv = rgb_to_yuv(rgb)
        centers = select_centers(
            xyz,
            args.patch_size,
            args.patch_stride,
            args.center_mode,
        )
        model_output, covered = run_models(
            models,
            xyz,
            yuv,
            centers,
            args,
            device,
        )
        out_yuv = np.zeros_like(yuv, dtype=np.float32)
        coverage_by_channel = {}
        for channel in range(3):
            out_yuv[:, channel] = compose_channel_prediction(
                yuv[:, channel],
                model_output[:, channel],
                covered,
                prediction_modes[channel],
                blends[channel],
                max_deltas[channel],
            )
            coverage_by_channel[CHANNEL_NAMES[channel]] = float(np.mean(covered))
        out_rgb = yuv_to_rgb(out_yuv)
        write_ply_xyz_rgb(out_path, xyz, out_rgb)
        record = {
            "input": str(frame_path),
            "output": str(out_path),
            "points": int(xyz.shape[0]),
            "patches": int(centers.shape[0]),
            "center_mode": args.center_mode,
            "prediction_modes": dict(zip(CHANNEL_NAMES, prediction_modes)),
            "blends": dict(zip(CHANNEL_NAMES, blends)),
            "max_deltas": dict(zip(CHANNEL_NAMES, max_deltas)),
            "coverage": coverage_by_channel,
            "seconds": round(time.time() - tic, 4),
        }
        print(record)
        if args.runtime_log:
            with args.runtime_log.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")


if __name__ == "__main__":
    main()
