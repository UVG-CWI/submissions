#!/usr/bin/env python3
"""Create a DQPC submission folder and zip archive."""

from __future__ import annotations

import argparse
import glob
import json
import platform
import shutil
import subprocess
import time
import zipfile
from pathlib import Path

import numpy as np


def relative_path(root: Path, path: Path) -> Path:
    try:
        return path.resolve().relative_to(root.resolve())
    except ValueError:
        return Path(path.name)


def validate_rgb_ply(path: Path) -> tuple[int, np.ndarray]:
    from dqpc_data import read_ply_xyz_rgb

    xyz, rgb = read_ply_xyz_rgb(path)
    if xyz.shape[0] == 0:
        raise ValueError(f"Final PLY is empty: {path}")
    if not np.all(np.isfinite(xyz)):
        raise ValueError(f"Final PLY contains non-finite XYZ values: {path}")
    if rgb is None:
        raise ValueError(f"Final PLY is missing RGB fields: {path}")
    if not np.all(np.isfinite(rgb)):
        raise ValueError(f"Final PLY contains non-finite RGB values: {path}")
    return int(xyz.shape[0]), xyz


def validate_quality_config(path: Path) -> dict:
    config = json.loads(path.read_text(encoding="utf-8"))
    if not bool(config.get("quality_gate_passed", False)):
        raise ValueError(f"Quality config did not pass its PSNR gate: {path}")
    drift_guard = config.get("color_drift_guard")
    if not isinstance(drift_guard, dict) or not bool(drift_guard.get("enabled", False)):
        raise ValueError(f"Quality config is missing the enabled color drift guard: {path}")
    return {
        "path": str(path),
        "quality_gate_passed": True,
        "selection_decision": config.get("selection_decision"),
        "candidate_used": config.get("candidate_used"),
        "color_drift_guard_enabled": True,
    }


def validate_frame_set(
    output_files: list[Path],
    output_root: Path,
    expected_glob: str,
    expected_root: Path,
    limit: int,
) -> dict:
    expected_files = sorted(Path(p) for p in glob.glob(expected_glob))
    if limit > 0:
        expected_files = expected_files[:limit]
    if not expected_files:
        raise RuntimeError(f"No expected source PLY files found from {expected_glob}")
    output_rel = {relative_path(output_root, path) for path in output_files}
    expected_rel = {relative_path(expected_root, path) for path in expected_files}
    missing = sorted(str(path) for path in expected_rel - output_rel)
    extra = sorted(str(path) for path in output_rel - expected_rel)
    if missing or extra:
        raise ValueError(
            "Submission frame set does not match the expected input frame set: "
            f"missing={missing[:10]} extra={extra[:10]}"
        )
    return {
        "expected_frame_count": len(expected_rel),
        "missing_frame_count": 0,
        "extra_frame_count": 0,
    }


def validate_geometry(
    output_path: Path,
    output_xyz: np.ndarray,
    output_root: Path,
    geometry_reference_root: Path,
    xyz_atol: float,
) -> None:
    from dqpc_data import read_ply_xyz_rgb

    rel = relative_path(output_root, output_path)
    reference_path = geometry_reference_root / rel
    if not reference_path.exists():
        raise FileNotFoundError(f"Missing geometry reference for {output_path}: {reference_path}")
    reference_xyz, _ = read_ply_xyz_rgb(reference_path)
    if reference_xyz.shape != output_xyz.shape:
        raise ValueError(
            f"Final PLY changed point count for {rel}: "
            f"{output_xyz.shape[0]} vs {reference_xyz.shape[0]}"
        )
    if not np.allclose(output_xyz, reference_xyz, rtol=0.0, atol=xyz_atol):
        raise ValueError(f"Final PLY changed XYZ or point order relative to geometry reference: {rel}")


def collect_hardware() -> dict:
    info = {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python": platform.python_version(),
    }
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda_available"] = bool(torch.cuda.is_available())
        info["cuda_device_count"] = int(torch.cuda.device_count())
        info["cuda_version"] = torch.version.cuda
        if torch.cuda.is_available():
            info["cuda_devices"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    except Exception as exc:
        info["torch_error"] = repr(exc)
    try:
        result = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"], capture_output=True, text=True, check=False)
        info["nvidia_smi"] = result.stdout.strip()
    except Exception as exc:
        info["nvidia_smi_error"] = repr(exc)
    return info


def read_jsonl(path: Path) -> list[dict]:
    records = []
    if not path.exists():
        return records
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_method_readme(path: Path, method_name: str) -> None:
    text = f"""# {method_name}

This submission uses a cascaded dynamic point cloud enhancement baseline:

1. PU-Dense sparse-convolution geometry enhancement.
2. Gaussian/DA-KNN color transfer from the consumer-grade colored point cloud.
3. Optional GQE-Net Y/U/V color refinement.
4. Optional lightweight temporal color smoothing.

External data: none, unless explicitly stated by the submitter.

Output format: one colored PLY per frame with x/y/z/red/green/blue vertex fields.
"""
    path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create DQPC submission archive")
    parser.add_argument("--input-root", required=True, type=Path, help="Root containing final colored PLY output")
    parser.add_argument("--ply-glob", required=True, help="Glob for final colored PLY files")
    parser.add_argument("--runtime-glob", default="outputs/dqpc_b4/runtime/*.jsonl")
    parser.add_argument("--out-dir", default=Path("outputs/dqpc_b4/submission"), type=Path)
    parser.add_argument("--zip-path", default=Path("outputs/dqpc_b4/submission.zip"), type=Path)
    parser.add_argument("--method-name", default="DQPC-B4-PUDense-DAKNN-GQENet")
    parser.add_argument(
        "--package-mode",
        default="ply-only",
        choices=("ply-only", "full"),
        help="ply-only zips final PLY files only; full also adds manifest, hardware, runtime, and README.",
    )
    parser.add_argument(
        "--validate-rgb",
        action="store_true",
        help="Read each PLY and verify RGB fields. This requires plyfile/numpy in the Python environment.",
    )
    parser.add_argument("--expected-root", default=None, type=Path)
    parser.add_argument("--expected-glob", default=None)
    parser.add_argument("--geometry-reference-root", default=None, type=Path)
    parser.add_argument("--quality-config", default=None, type=Path)
    parser.add_argument("--xyz-atol", default=1e-4, type=float)
    parser.add_argument("--limit", default=0, type=int)
    args = parser.parse_args()

    tic = time.time()
    ply_files = sorted(Path(p) for p in glob.glob(args.ply_glob))
    if args.limit > 0:
        ply_files = ply_files[: args.limit]
    if not ply_files:
        raise RuntimeError(f"No final PLY files found from {args.ply_glob}")
    if (args.expected_root is None) != (args.expected_glob is None):
        raise ValueError("--expected-root and --expected-glob must be provided together")
    validation = {}
    if args.expected_root is not None and args.expected_glob is not None:
        validation.update(
            validate_frame_set(
                ply_files,
                args.input_root,
                args.expected_glob,
                args.expected_root,
                args.limit,
            )
        )
    if args.quality_config is not None:
        if not args.quality_config.exists():
            raise FileNotFoundError(f"Missing quality config: {args.quality_config}")
        validation["quality_config"] = validate_quality_config(args.quality_config)

    if args.out_dir.exists():
        shutil.rmtree(args.out_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = args.out_dir / "frames" if args.package_mode == "full" else args.out_dir
    frames_dir.mkdir(parents=True, exist_ok=True)

    manifest_frames = []
    for ply in ply_files:
        point_count = None
        xyz = None
        if args.validate_rgb or args.geometry_reference_root is not None:
            point_count, xyz = validate_rgb_ply(ply)
        if args.geometry_reference_root is not None:
            assert xyz is not None
            validate_geometry(
                ply,
                xyz,
                args.input_root,
                args.geometry_reference_root,
                args.xyz_atol,
            )
        rel = relative_path(args.input_root, ply)
        dst = frames_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ply, dst)
        frame_record = {"source": str(ply), "path": str(dst.relative_to(args.out_dir))}
        if point_count is not None:
            frame_record["points"] = point_count
        manifest_frames.append(frame_record)

    if args.package_mode == "full":
        runtime_records = []
        from summarize_validation import collect_runtime

        runtime_dir = args.out_dir / "runtime"
        runtime_dir.mkdir(parents=True, exist_ok=True)
        for runtime_path in sorted(Path(p) for p in glob.glob(args.runtime_glob)):
            shutil.copy2(runtime_path, runtime_dir / runtime_path.name)
            runtime_records.extend(read_jsonl(runtime_path))

        runtime_summary = collect_runtime(args.runtime_glob)
        manifest = {
            "method": args.method_name,
            "created_unix": time.time(),
            "frame_count": len(manifest_frames),
            "frames": manifest_frames,
            "validation": validation,
            "geometry_reference_root": (
                str(args.geometry_reference_root) if args.geometry_reference_root else None
            ),
            "runtime_log_count": len(runtime_records),
            "runtime_summary": runtime_summary,
            "runtime_seconds_sum": runtime_summary.get("runtime_seconds_sum", 0.0),
            "runtime_seconds_mean": runtime_summary.get("runtime_seconds_mean"),
        }
        (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        (args.out_dir / "hardware.json").write_text(json.dumps(collect_hardware(), indent=2), encoding="utf-8")
        write_method_readme(args.out_dir / "README.md", args.method_name)

    args.zip_path.parent.mkdir(parents=True, exist_ok=True)
    if args.zip_path.exists():
        args.zip_path.unlink()
    with zipfile.ZipFile(args.zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(args.out_dir.rglob("*")):
            if path.is_file():
                zf.write(path, path.relative_to(args.out_dir))

    print(
        json.dumps(
            {
                "out_dir": str(args.out_dir),
                "zip_path": str(args.zip_path),
                "frames": len(manifest_frames),
            "package_mode": args.package_mode,
            "validate_rgb": bool(args.validate_rgb),
            "validation": validation,
            "geometry_reference_root": (
                str(args.geometry_reference_root) if args.geometry_reference_root else None
            ),
            "seconds": round(time.time() - tic, 4),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
