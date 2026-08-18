#!/usr/bin/env python3
"""Train/fine-tune PU-Dense geometry on DQPC CG_15 -> HE_15 pairs."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from dqpc_data import DQPCPairDataset, collate_sparse_batch, discover_frame_pairs, infer_voxel_size_from_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DQPC PU-Dense geometry training")
    parser.add_argument("--pudense-root", default="external/PointCloudUpsampling", type=Path)
    parser.add_argument("--cg-glob", required=True, help="Glob for CG train PLY files")
    parser.add_argument("--he-glob", default=None, help="Optional glob for HE train PLY files")
    parser.add_argument("--cg-token", default="/CG/15fps/")
    parser.add_argument("--he-token", default="/HE/15fps/")
    parser.add_argument(
        "--voxel-size",
        default="auto",
        help="Quantization voxel size. Use 'auto' to choose 1.0 for mm-like coordinates or 0.001 for meter-like coordinates.",
    )
    parser.add_argument("--crop-size", default=256, type=int, help="Training crop size in quantized voxels; 0 disables crop")
    parser.add_argument("--max-cg-points", default=70000, type=int)
    parser.add_argument("--max-he-points", default=280000, type=int)
    parser.add_argument("--batch-size", default=1, type=int)
    parser.add_argument("--num-workers", default=0, type=int)
    parser.add_argument("--steps", default=2000, type=int)
    parser.add_argument("--lr", default=8e-4, type=float)
    parser.add_argument("--save-every", default=500, type=int)
    parser.add_argument("--last-kernel-size", default=5, type=int)
    parser.add_argument("--init-ckpt", default=None, type=Path)
    parser.add_argument("--out-dir", default=Path("outputs/dqpc_b4/pudense_ckpts"), type=Path)
    parser.add_argument("--seed", default=1, type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sys.path.insert(0, str(args.pudense_root.resolve()))

    import MinkowskiEngine as ME
    from model.Network import MyNet
    from utils.loss import get_metrics

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    pairs = discover_frame_pairs(args.cg_glob, args.he_glob, args.cg_token, args.he_token, require_gt=True)
    if not pairs:
        raise RuntimeError(f"No training pairs found from {args.cg_glob}")
    voxel_size = infer_voxel_size_from_paths([p.cg_path for p in pairs], args.voxel_size)
    print({"resolved_voxel_size": voxel_size, "voxel_size_arg": str(args.voxel_size), "pair_count": len(pairs)})

    dataset = DQPCPairDataset(
        pairs,
        voxel_size=voxel_size,
        crop_size=args.crop_size,
        max_cg_points=args.max_cg_points,
        max_he_points=args.max_he_points,
        seed=args.seed,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_sparse_batch,
        drop_last=False,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MyNet(last_kernel_size=args.last_kernel_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=1e-4)
    start_step = 1
    if args.init_ckpt:
        ckpt = torch.load(args.init_ckpt, map_location=device)
        model.load_state_dict(ckpt["model"])
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        start_step = int(ckpt.get("step", 0)) + 1

    args.out_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.out_dir / "train_log.jsonl"
    criterion = torch.nn.BCEWithLogitsLoss()
    data_iter = iter(loader)
    model.train()

    for step in range(start_step, args.steps + 1):
        try:
            coords, feats, coords_t, batch_pairs = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            coords, feats, coords_t, batch_pairs = next(data_iter)

        tic = time.time()
        optimizer.zero_grad(set_to_none=True)
        x = ME.SparseTensor(feats=feats, coords=coords).to(device)
        _, out_cls, target, keep = model(x, coords_T=coords_t, device=device, prune=False)
        loss = criterion(out_cls.F.squeeze(), target.type(out_cls.F.dtype).to(device))
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"skip step {step}: invalid loss {loss.item()}")
            continue
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        metrics = get_metrics(keep, target)
        record = {
            "step": step,
            "loss": float(loss.item()),
            "precision": float(metrics[0]),
            "recall": float(metrics[1]),
            "iou": float(metrics[2]),
            "seconds": round(time.time() - tic, 4),
            "frames": [str(p.cg_path) for p in batch_pairs],
        }
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
        if step == 1 or step % 20 == 0:
            print(record)

        if step % args.save_every == 0 or step == args.steps:
            ckpt_path = args.out_dir / f"iter{step}.pth"
            torch.save({"step": step, "model": model.state_dict(), "optimizer": optimizer.state_dict()}, ckpt_path)
            print(f"saved {ckpt_path}")


if __name__ == "__main__":
    main()
