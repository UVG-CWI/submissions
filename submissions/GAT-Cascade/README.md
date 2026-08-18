# GAT-Cascade

Geometry-Attribute-Temporal Cascaded Enhancement for Dual-Quality Dynamic Point Clouds

## Team

GAT-Cascade

## Algorithm Name

GAT-Cascade: Geometry-Attribute-Temporal Cascaded Enhancement for Dual-Quality Dynamic Point Clouds.

## Team Members

| Name | Email | Affiliation |
|---|---|---|
| Liang Xie | lxie5201@outlook.com | School of Computer Science and Technology, Guangdong University of Technology, Guangzhou, China |
| Le Wang | wangle2@mails.gdut.edu.cn | School of Computer Science and Technology, Guangdong University of Technology, Guangzhou, China |
| Songlin Fan | slfan@pku.edu.cn | Institute of Trustworthy Embodied Artificial Intelligence, Fudan University, Shanghai, China |
| Kaihong Yu | ykh2204@outlook.com | School of Computer Science and Technology, Guangdong University of Technology, Guangzhou, China |
| Jianguo Zhang | zhangjianguo@cuhk.edu.cn | Shenzhen Institute of Artificial Intelligence and Robotics for Society, Shenzhen, China |
| Wei Gao | gaowei262@pku.edu.cn | Guangdong Provincial Key Laboratory of Ultra High Definition Immersive Media Technology, Shenzhen Graduate School, Peking University, Shenzhen, China |

## Track

UVG-CWI-DQPC Enhancement Only track.

## Method

GAT-Cascade enhances consumer-grade dynamic colored point clouds with a conservative cascade:

1. PU-Dense sparse-convolution geometry enhancement.
2. Gaussian or DA-KNN RGB transfer from the input consumer-grade frame to the enhanced geometry.
3. GQE-Net Y/U/V local color refinement with validation-calibrated channel blending.
4. Lightweight temporal color stabilization through reliable adjacent-frame correspondences.

The implementation is under `baselines/dqpc_b4/`. External baseline code used by the pipeline is kept under `external/`.

## Input And Output

The runner expects the UVG-CWI-DQPC dataset layout:

```text
$DQPC_ROOT/
  train/<sequence>/consumer-grade_capture_system/CG_aligned/15fps/*.ply
  train/<sequence>/high-end_capture_system/HE/15fps/*.ply
  valid/<sequence>/consumer-grade_capture_system/CG_aligned/15fps/*.ply
  valid/<sequence>/high-end_capture_system/HE/15fps/*.ply
  test/<sequence>/consumer-grade_capture_system/CG_aligned/15fps/*.ply
```

Compact layouts such as `CG_aligned/15fps`, `CGv2/15fps`, `CGv2_15`, and `CG/15fps` are auto-detected. The output is a zip archive containing one RGB PLY file per input frame with `x`, `y`, `z`, `red`, `green`, and `blue` vertex fields. Input PLY files and generated output PLY files are not included in this repository.

## Environment

The geometry stage uses PU-Dense and MinkowskiEngine, so the recommended environment is:

```text
Ubuntu 20.04
CUDA 11.x
Python 3.8
PyTorch 1.7.x
MinkowskiEngine 0.4.x
```

Python package dependencies are listed in `requirements.txt`. A Docker recipe is provided in `Dockerfile`.

## Quick Start

Build the Docker image:

```bash
docker build -t gat-cascade .
```

Run inference on the test split:

```bash
docker run --gpus all --rm \
  -v /path/to/UVG-CWI-DQPC:/data/dqpc:ro \
  -v /path/to/checkpoints:/checkpoints:ro \
  -v "$PWD/outputs":/workspace/outputs \
  -e DQPC_ROOT=/data/dqpc \
  -e SPLIT=test \
  -e PUDENSE_CKPT=/checkpoints/pudense.pth \
  -e MODEL_Y=/checkpoints/gqenet_y.pth \
  -e MODEL_U=/checkpoints/gqenet_u.pth \
  -e MODEL_V=/checkpoints/gqenet_v.pth \
  gat-cascade
```

Without Docker, run the same entry point from the repository root:

```bash
export DQPC_ROOT=/path/to/UVG-CWI-DQPC
export SPLIT=test
export PUDENSE_CKPT=/path/to/pudense.pth
export MODEL_Y=/path/to/gqenet_y.pth
export MODEL_U=/path/to/gqenet_u.pth
export MODEL_V=/path/to/gqenet_v.pth
bash run.sh
```

The default final archive is written to:

```text
outputs/dqpc_b4/submission_<split>.zip
```

## Runtime

The runtime depends on frame count, point count, GPU model, and whether the optional GQE-Net and temporal stages are enabled. The runner records per-frame JSONL timing logs in `outputs/dqpc_b4/runtime/`. For official execution, use `PACKAGE_MODE=ply-only`; for internal timing diagnostics, use `PACKAGE_MODE=full`.

## Runtime Options

Useful environment variables:

| Variable | Default | Description |
|---|---:|---|
| `PYTHON_BIN` | `python` | Python interpreter inside the configured environment. |
| `DQPC_ROOT` | required | Dataset root. |
| `SPLIT` | `test` | Dataset split to process. |
| `PUDENSE_CKPT` | required | PU-Dense geometry checkpoint. |
| `MODEL_Y`, `MODEL_U`, `MODEL_V` | optional | GQE-Net Y/U/V checkpoints. If all are set, color refinement is run. |
| `APPLY_GQENET` | `auto` | `auto`, `1`, or `0`. |
| `APPLY_TEMPORAL` | `1` | Apply temporal color smoothing. |
| `CONFIG_IN` | empty | Validation blend JSON used when applying calibrated GQE-Net blending to test data. |
| `PACKAGE_MODE` | `ply-only` | `ply-only` for official submission archive, `full` for internal diagnostics. |
| `LIMIT` | `0` | Limit frame count for smoke tests; `0` means all frames. |

For a one-frame smoke test:

```bash
export LIMIT=1
bash run.sh
```

## Reproducibility Notes

The paper description is provided in `Description.txt`; the camera-ready paper is `CR__Geometry_Attribute_Temporal_Cascaded_Enhancement_for_Dual_Quality_Dynamic_Point_Clouds.pdf`. On the local validation split, GAT-Cascade reports a loose-threshold F-score of 0.8519 and six-view projection SSIM of 0.9184 over 564 frames.

The submission runner logs runtime JSONL files in `outputs/dqpc_b4/runtime/` and uses `baselines/dqpc_b4/scripts/07_make_submission.sh` to validate RGB PLY files and package the final archive.
