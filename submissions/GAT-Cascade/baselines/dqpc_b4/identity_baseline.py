#!/usr/bin/env python3
"""Copy CG PLY files into a baseline output tree and record runtime."""

from __future__ import annotations

import argparse
import glob
import json
import shutil
import time
from pathlib import Path

from gqenet_dqpc import relative_path


def main() -> None:
    parser = argparse.ArgumentParser(description="DQPC identity baseline")
    parser.add_argument("--input-root", required=True, type=Path)
    parser.add_argument("--cg-glob", required=True)
    parser.add_argument("--out-root", required=True, type=Path)
    parser.add_argument("--runtime-log", default=None, type=Path)
    parser.add_argument("--limit", default=0, type=int)
    args = parser.parse_args()

    frames = sorted(Path(p) for p in glob.glob(args.cg_glob))
    if args.limit > 0:
        frames = frames[: args.limit]
    if not frames:
        raise RuntimeError(f"No CG frames found from {args.cg_glob}")
    if args.runtime_log:
        args.runtime_log.parent.mkdir(parents=True, exist_ok=True)

    for frame in frames:
        tic = time.time()
        out = args.out_root / relative_path(args.input_root, frame)
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(frame, out)
        record = {"input": str(frame), "output": str(out), "seconds": round(time.time() - tic, 6)}
        print(record)
        if args.runtime_log:
            with args.runtime_log.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")


if __name__ == "__main__":
    main()
