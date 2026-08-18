#!/usr/bin/env python3
"""Optional wrappers for external point-cloud metrics such as pc_error and PCQM."""

from __future__ import annotations

import argparse
import glob
import json
import math
import re
import shutil
import subprocess
from pathlib import Path


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


PCQM_RESOURCE_FILES = (
    "L_data.txt",
    "RegularGrid_0_0_1.txt",
    "RegularGrid_0_0_2.txt",
    "RegularGridInit_0_0_1.txt",
    "RegularGridInit_0_0_2.txt",
)


def relative_path(root: Path, path: Path) -> Path:
    try:
        return path.resolve().relative_to(root.resolve())
    except ValueError:
        return Path(path.name)


def run_cmd(cmd: list[str], cwd: Path | None = None) -> dict:
    result = subprocess.run(cmd, cwd=str(cwd) if cwd else None, capture_output=True, text=True, check=False)
    record = {"cmd": cmd, "returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr}
    if cwd:
        record["cwd"] = str(cwd)
    return record


def parse_pcqm_value(text: str) -> float | None:
    match = re.search(r"PCQM\s+value\s+is\s*:\s*([-+0-9.eE]+)", text)
    if not match:
        return None
    return float(match.group(1))


def prepare_pcqm_work_dir(pcqm_bin: Path, work_dir: Path) -> Path:
    work_dir.mkdir(parents=True, exist_ok=True)
    source_dir = pcqm_bin.resolve().parent
    for name in PCQM_RESOURCE_FILES:
        src = source_dir / name
        dst = work_dir / name
        if src.exists() and not dst.exists():
            shutil.copy2(src, dst)
    return work_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Run optional external metrics for DQPC outputs")
    parser.add_argument("--pred-root", required=True, type=Path)
    parser.add_argument("--pred-glob", required=True)
    parser.add_argument("--gt-root", required=True, type=Path)
    parser.add_argument("--out-jsonl", required=True, type=Path)
    parser.add_argument("--pc-error-bin", default=None, type=Path)
    parser.add_argument("--pc-error-res", default="1024")
    parser.add_argument("--pcqm-bin", default=None, type=Path)
    parser.add_argument("--pcqm-work-dir", default=None, type=Path)
    parser.add_argument("--summary-json", default=None, type=Path)
    parser.add_argument("--limit", default=0, type=int)
    args = parser.parse_args()

    preds = sorted(Path(p) for p in glob.glob(args.pred_glob))
    if args.limit > 0:
        preds = preds[: args.limit]
    if not preds:
        raise RuntimeError(f"No prediction files found from {args.pred_glob}")
    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    pcqm_work_dir = None
    if args.pcqm_bin:
        pcqm_work_dir = prepare_pcqm_work_dir(args.pcqm_bin, args.pcqm_work_dir or args.out_jsonl.parent / "pcqm_work")

    records = []
    with args.out_jsonl.open("w", encoding="utf-8") as f:
        for pred in preds:
            rel = relative_path(args.pred_root, pred)
            gt = matching_he_path(args.gt_root, rel) or (args.gt_root / rel)
            if not gt.exists():
                raise FileNotFoundError(f"Missing GT for {pred}: {gt}")
            record = {"pred": str(pred), "gt": str(gt), "metrics": []}
            if args.pc_error_bin:
                record["metrics"].append(
                    run_cmd([str(args.pc_error_bin), "-a", str(gt), "-b", str(pred), "--resolution", str(args.pc_error_res)])
                )
            if args.pcqm_bin:
                pcqm_record = run_cmd(
                    [str(args.pcqm_bin), str(gt.resolve()), str(pred.resolve()), "--fastquit"],
                    cwd=pcqm_work_dir,
                )
                pcqm_value = parse_pcqm_value(pcqm_record["stdout"])
                if pcqm_value is not None:
                    record["pcqm"] = pcqm_value
                record["metrics"].append(pcqm_record)
            records.append(record)
            f.write(json.dumps(record) + "\n")
            print(json.dumps({"pred": str(pred), "metric_count": len(record["metrics"])}))

    if args.summary_json:
        summary = {"count": len(records)}
        pcqm_values = [float(r["pcqm"]) for r in records if "pcqm" in r and math.isfinite(float(r["pcqm"]))]
        if pcqm_values:
            summary["mean_pcqm"] = float(sum(pcqm_values) / len(pcqm_values))
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
