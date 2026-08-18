#!/usr/bin/env python3
"""Train GQE-Net Y/U/V color refinement models on DQPC enhanced frames."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dqpc_data import distance_scale_from_paths
from gqenet_dqpc import DQPCGQEDataset, discover_gqe_pairs


CHANNEL_NAMES = ["y", "u", "v"]


def checkpoint_prediction_mode(ckpt: dict) -> str:
    mode = ckpt.get("prediction_mode")
    if mode in {"absolute", "residual"}:
        return str(mode)
    if "residual" in ckpt:
        return "residual" if bool(ckpt["residual"]) else "absolute"
    return "absolute"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train GQE-Net on DQPC patches")
    parser.add_argument("--gqenet-root", default="external/GQE-Net", type=Path)
    parser.add_argument("--enhanced-root", required=True, type=Path)
    parser.add_argument("--enhanced-glob", required=True)
    parser.add_argument("--he-root", required=True, type=Path)
    parser.add_argument("--channel", required=True, type=int, choices=[0, 1, 2])
    parser.add_argument("--patch-size", default=2048, type=int)
    parser.add_argument("--samples-per-epoch", default=10000, type=int)
    parser.add_argument("--batch-size", default=8, type=int)
    parser.add_argument("--epochs", default=20, type=int)
    parser.add_argument("--lr", default=2.5e-3, type=float)
    parser.add_argument("--coord-scale", default=1.0, type=float)
    parser.add_argument("--residual", action="store_true", help="Train GQE-Net to predict target-input residuals instead of absolute YUV")
    parser.add_argument("--target-max-distance", default=20.0, type=float, help="Down-weight HE labels farther than this distance; interpreted by --distance-unit")
    parser.add_argument("--distance-unit", choices=("mm", "coordinate"), default="mm")
    parser.add_argument("--far-target-weight", default=0.2, type=float)
    parser.add_argument("--max-cached-frames", default=8, type=int)
    parser.add_argument(
        "--frame-sampling",
        choices=("v3_random", "cache_local"),
        default="v3_random",
        help="v3_random preserves V3 sampling; cache_local groups patches by frame to reduce PLY/KD-tree reloads.",
    )
    parser.add_argument("--patches-per-frame", default=8, type=int)
    parser.add_argument("--val-enhanced-root", default=None, type=Path)
    parser.add_argument("--val-enhanced-glob", default=None)
    parser.add_argument("--val-he-root", default=None, type=Path)
    parser.add_argument("--val-samples", default=512, type=int)
    parser.add_argument("--num-workers", default=0, type=int)
    parser.add_argument("--out-dir", default=Path("outputs/dqpc_b4/gqenet_ckpts"), type=Path)
    parser.add_argument("--init-ckpt", default=None, type=Path)
    parser.add_argument("--seed", default=1, type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sys.path.insert(0, str(args.gqenet_root.resolve()))
    import model_GQE_Net_final as gqe_model

    from model_GQE_Net_final import GAPCN

    torch.manual_seed(args.seed)
    pairs = discover_gqe_pairs(args.enhanced_glob, args.enhanced_root, args.he_root)
    if not pairs:
        raise RuntimeError(f"No enhanced frames found from {args.enhanced_glob}")
    distance_scale = distance_scale_from_paths([pair.enhanced_path for pair in pairs], args.distance_unit)
    target_max_distance = args.target_max_distance * distance_scale

    dataset = DQPCGQEDataset(
        pairs,
        channel=args.channel,
        patch_size=args.patch_size,
        samples_per_epoch=args.samples_per_epoch,
        coord_scale=args.coord_scale,
        residual=args.residual,
        target_max_distance=target_max_distance,
        far_target_weight=args.far_target_weight,
        max_cached_frames=args.max_cached_frames,
        frame_sampling=args.frame_sampling,
        patches_per_frame=args.patches_per_frame,
        seed=args.seed,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=True,
        pin_memory=device.type == "cuda",
    )
    val_args = (args.val_enhanced_root, args.val_enhanced_glob, args.val_he_root)
    if any(value is not None for value in val_args) and not all(value is not None for value in val_args):
        raise ValueError("--val-enhanced-root, --val-enhanced-glob, and --val-he-root must be provided together")
    val_batches = []
    if all(value is not None for value in val_args):
        val_pairs = discover_gqe_pairs(
            args.val_enhanced_glob,
            args.val_enhanced_root,
            args.val_he_root,
        )
        if not val_pairs:
            raise RuntimeError(f"No validation enhanced frames found from {args.val_enhanced_glob}")
        val_distance_scale = distance_scale_from_paths(
            [pair.enhanced_path for pair in val_pairs],
            args.distance_unit,
        )
        val_dataset = DQPCGQEDataset(
            val_pairs,
            channel=args.channel,
            patch_size=args.patch_size,
            samples_per_epoch=args.val_samples,
            coord_scale=args.coord_scale,
            residual=args.residual,
            target_max_distance=args.target_max_distance * val_distance_scale,
            far_target_weight=args.far_target_weight,
            max_cached_frames=args.max_cached_frames,
            frame_sampling="cache_local",
            patches_per_frame=args.patches_per_frame,
            seed=args.seed + 1000003,
        )
        val_dataset.set_epoch(0)
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
            drop_last=False,
            pin_memory=False,
        )
        val_batches = list(val_loader)
    gqe_model.devices = str(device)
    model = GAPCN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    start_epoch = 0
    best_val_loss = float("inf")
    if args.init_ckpt:
        ckpt = torch.load(args.init_ckpt, map_location=device)
        requested_mode = "residual" if args.residual else "absolute"
        init_mode = checkpoint_prediction_mode(ckpt)
        if init_mode != requested_mode:
            raise ValueError(
                f"Checkpoint {args.init_ckpt} uses {init_mode} prediction, but this run requests "
                f"{requested_mode}. Do not initialize residual training from an absolute-output checkpoint."
            )
        if "channel" in ckpt and int(ckpt["channel"]) != args.channel:
            raise ValueError(
                f"Checkpoint {args.init_ckpt} is for channel {ckpt['channel']}, "
                f"but this run trains channel {args.channel}"
            )
        model.load_state_dict(ckpt["model_state_dict"])
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = int(ckpt.get("epoch", -1)) + 1
        best_val_loss = float(ckpt.get("best_val_loss", best_val_loss))

    channel_name = CHANNEL_NAMES[args.channel]
    out_dir = args.out_dir / channel_name
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "train_log.jsonl"
    for epoch in range(start_epoch, args.epochs):
        dataset.set_epoch(epoch)
        tic = time.time()
        model.train()
        total_loss = 0.0
        count = 0
        for data, label, weight in loader:
            data = data.to(device, non_blocking=True).permute(0, 2, 1).contiguous()
            label = label.to(device, non_blocking=True)
            weight = weight.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            pred = model(data)
            loss = ((pred - label) ** 2 * weight).sum() / weight.sum().clamp_min(1.0)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * data.shape[0]
            count += data.shape[0]
        avg_loss = total_loss / max(count, 1)
        val_loss = None
        val_psnr = None
        if val_batches:
            model.eval()
            val_loss_sum = 0.0
            val_count = 0
            with torch.no_grad():
                for data, label, weight in val_batches:
                    data = data.to(device, non_blocking=True).permute(0, 2, 1).contiguous()
                    label = label.to(device, non_blocking=True)
                    weight = weight.to(device, non_blocking=True)
                    pred = model(data)
                    batch_loss = ((pred - label) ** 2 * weight).sum() / weight.sum().clamp_min(1.0)
                    val_loss_sum += float(batch_loss.item()) * data.shape[0]
                    val_count += data.shape[0]
            val_loss = val_loss_sum / max(val_count, 1)
            val_psnr = float("inf") if val_loss <= 0 else 10.0 * math.log10((255.0 * 255.0) / val_loss)
        record = {
            "epoch": epoch,
            "channel": channel_name,
            "loss": avg_loss,
            "residual": bool(args.residual),
            "target_max_distance": float(args.target_max_distance),
            "distance_unit": args.distance_unit,
            "resolved_target_max_distance": float(target_max_distance),
            "far_target_weight": float(args.far_target_weight),
            "frame_sampling": args.frame_sampling,
            "patches_per_frame": int(args.patches_per_frame),
            "val_loss": val_loss,
            "val_psnr": val_psnr,
            "seconds": round(time.time() - tic, 4),
        }
        print(record)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
        ckpt_path = out_dir / f"model_{epoch}.pth"
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": avg_loss,
            "val_loss": val_loss,
            "val_psnr": val_psnr,
            "best_val_loss": min(best_val_loss, val_loss) if val_loss is not None else best_val_loss,
            "channel": args.channel,
            "patch_size": args.patch_size,
            "coord_scale": args.coord_scale,
            "residual": bool(args.residual),
            "prediction_mode": "residual" if args.residual else "absolute",
            "target_max_distance": float(args.target_max_distance),
            "distance_unit": args.distance_unit,
            "resolved_target_max_distance": float(target_max_distance),
            "far_target_weight": float(args.far_target_weight),
            "frame_sampling": args.frame_sampling,
            "patches_per_frame": int(args.patches_per_frame),
        }
        torch.save(checkpoint, ckpt_path)
        print(f"saved {ckpt_path}")
        if val_loss is not None and val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint["best_val_loss"] = best_val_loss
            best_path = out_dir / "best.pth"
            torch.save(checkpoint, best_path)
            print(f"saved {best_path}")


if __name__ == "__main__":
    main()
