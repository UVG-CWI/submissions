#!/usr/bin/env python3
"""Batch Gaussian KNN recoloring for DQPC sequence outputs."""

from __future__ import annotations

import argparse
import glob
import json
import time
from pathlib import Path

from dqpc_data import distance_scale_from_paths, read_ply_xyz, read_ply_xyz_rgb, write_ply_xyz_rgb
from recolor_gaussian import gaussian_recolor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Recolor PU-Dense geometry outputs from matching CG frames")
    parser.add_argument("--cg-glob", required=True)
    parser.add_argument("--input-root", required=True, type=Path)
    parser.add_argument("--geometry-root", required=True, type=Path)
    parser.add_argument("--colored-out-root", required=True, type=Path)
    parser.add_argument("--runtime-log", default=None, type=Path)
    parser.add_argument("--k", default=8, type=int)
    parser.add_argument("--sigma", default=None, type=float)
    parser.add_argument("--copy-distance", default=0.5, type=float)
    parser.add_argument("--distance-unit", choices=("mm", "coordinate"), default="mm")
    parser.add_argument("--query-chunk-size", default=250000, type=int)
    parser.add_argument("--limit", default=0, type=int)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--ascii", action="store_true")
    return parser.parse_args()


def rel_path(input_root: Path, frame: Path) -> Path:
    try:
        return frame.resolve().relative_to(input_root.resolve())
    except ValueError:
        return Path(frame.name)


def main() -> None:
    args = parse_args()
    frames = sorted(Path(p) for p in glob.glob(args.cg_glob))
    if args.limit > 0:
        frames = frames[: args.limit]
    if not frames:
        raise RuntimeError(f"No CG frames found from {args.cg_glob}")
    if args.runtime_log:
        args.runtime_log.parent.mkdir(parents=True, exist_ok=True)
    distance_scale = distance_scale_from_paths(frames, args.distance_unit)

    for idx, cg_path in enumerate(frames, start=1):
        tic = time.time()
        rel = rel_path(args.input_root, cg_path)
        geom_path = args.geometry_root / rel
        out_path = args.colored_out_root / rel
        if args.skip_existing and out_path.exists():
            print(f"[{idx}/{len(frames)}] skip existing {out_path}")
            continue
        if not geom_path.exists():
            raise FileNotFoundError(f"Missing geometry output for {cg_path}: {geom_path}")

        source_xyz, source_rgb = read_ply_xyz_rgb(cg_path)
        if source_rgb is None:
            raise ValueError(f"{cg_path} has no RGB fields")
        target_xyz = read_ply_xyz(geom_path)
        copy_distance = args.copy_distance * distance_scale if args.copy_distance >= 0 else None
        target_rgb = gaussian_recolor(
            source_xyz,
            source_rgb,
            target_xyz,
            args.k,
            args.sigma,
            copy_distance=copy_distance,
            query_chunk_size=args.query_chunk_size,
        )
        write_ply_xyz_rgb(out_path, target_xyz, target_rgb, text=args.ascii)

        record = {
            "source": str(cg_path),
            "geometry": str(geom_path),
            "output": str(out_path),
            "source_points": int(source_xyz.shape[0]),
            "output_points": int(target_xyz.shape[0]),
            "copy_distance": args.copy_distance,
            "distance_unit": args.distance_unit,
            "resolved_copy_distance": copy_distance,
            "seconds": round(time.time() - tic, 4),
        }
        print(record)
        if args.runtime_log:
            with args.runtime_log.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")


if __name__ == "__main__":
    main()
