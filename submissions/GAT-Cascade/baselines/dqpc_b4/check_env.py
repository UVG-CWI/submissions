#!/usr/bin/env python3
"""Check the Python environment required by the DQPC B4 baseline."""

from __future__ import annotations

import argparse
import importlib.util
import json
import platform
import sys
from pathlib import Path


def module_status(name: str) -> dict:
    spec = importlib.util.find_spec(name)
    status = {"available": spec is not None, "version": None, "path": None}
    if spec is None:
        return status
    try:
        module = __import__(name)
        status["version"] = getattr(module, "__version__", None)
        status["path"] = getattr(module, "__file__", None)
    except Exception as exc:
        status["error"] = repr(exc)
    return status


def main() -> None:
    parser = argparse.ArgumentParser(description="Check DQPC B4 baseline runtime environment")
    parser.add_argument("--json-out", default=None, type=Path)
    args = parser.parse_args()

    modules = ["numpy", "scipy", "plyfile", "torch", "MinkowskiEngine"]
    report = {
        "python": sys.executable,
        "python_version": sys.version,
        "platform": platform.platform(),
        "modules": {name: module_status(name) for name in modules},
    }
    if report["modules"]["torch"]["available"]:
        try:
            import torch

            report["torch_cuda_available"] = bool(torch.cuda.is_available())
            report["torch_cuda_device_count"] = int(torch.cuda.device_count())
            report["torch_cuda_version"] = torch.version.cuda
        except Exception as exc:
            report["torch_cuda_error"] = repr(exc)

    print(json.dumps(report, indent=2))
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    missing = [name for name, item in report["modules"].items() if not item["available"]]
    if missing:
        raise SystemExit(f"Missing required modules: {', '.join(missing)}")


if __name__ == "__main__":
    main()
