#!/usr/bin/env python3
"""Check DQPC directory layout and PLY fields."""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

from dqpc_data import default_cg_glob, discover_frame_pairs, infer_voxel_size_from_paths, read_ply_xyz_rgb


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check DQPC dataset layout for B4 baseline")
    parser.add_argument("--dqpc-root", required=True, type=Path)
    parser.add_argument("--split", default="train")
    parser.add_argument("--cg-glob", default=None)
    parser.add_argument("--he-glob", default=None)
    parser.add_argument("--require-gt", action="store_true")
    parser.add_argument("--voxel-size", default="auto")
    parser.add_argument("--max-check", default=5, type=int)
    parser.add_argument("--json-out", default=None, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cg_glob = args.cg_glob or default_cg_glob(args.dqpc_root, args.split)
    he_glob = args.he_glob
    cg_files = sorted(glob.glob(cg_glob))
    pairs = discover_frame_pairs(cg_glob, he_glob=he_glob, require_gt=args.require_gt)

    checked = []
    for pair in pairs[: args.max_check]:
        cg_xyz, cg_rgb = read_ply_xyz_rgb(pair.cg_path)
        item = {
            "cg": str(pair.cg_path),
            "he": str(pair.he_path) if pair.he_path else None,
            "sequence": pair.sequence,
            "frame_id": pair.frame_id,
            "cg_points": int(cg_xyz.shape[0]),
            "cg_has_rgb": cg_rgb is not None,
        }
        if pair.he_path:
            he_xyz, he_rgb = read_ply_xyz_rgb(pair.he_path)
            item["he_points"] = int(he_xyz.shape[0])
            item["he_has_rgb"] = he_rgb is not None
        checked.append(item)

    report = {
        "dqpc_root": str(args.dqpc_root),
        "split": args.split,
        "cg_glob": cg_glob,
        "he_glob": he_glob,
        "resolved_voxel_size": infer_voxel_size_from_paths([p.cg_path for p in pairs], args.voxel_size) if pairs else None,
        "cg_file_count": len(cg_files),
        "pair_count": len(pairs),
        "checked": checked,
    }
    print(json.dumps(report, indent=2))
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    if not cg_files:
        raise SystemExit(f"No CG PLY files found: {cg_glob}")
    if args.require_gt and len(pairs) != len(cg_files):
        raise SystemExit("Some CG frames are missing HE pairs")
    if any(not item["cg_has_rgb"] for item in checked):
        raise SystemExit("At least one checked CG PLY is missing RGB fields")


if __name__ == "__main__":
    main()
