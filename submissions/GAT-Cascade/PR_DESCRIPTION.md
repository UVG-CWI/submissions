# Submission: GAT-Cascade

## Summary

This PR adds the GAT-Cascade submission for the UVG-CWI-DQPC Enhancement Only track.

GAT-Cascade is a geometry-attribute-temporal cascaded enhancement pipeline for consumer-grade dynamic colored point clouds. It combines PU-Dense sparse-convolution geometry recovery, deterministic Gaussian/DA-KNN color transfer, GQE-Net Y/U/V color refinement with validation-calibrated blending, and lightweight temporal color stabilization.

## Files

- `README.md`: team information, method summary, environment, and execution instructions.
- `Dockerfile`: CUDA/PyTorch/MinkowskiEngine environment for running the pipeline.
- `requirements.txt`: Python package dependencies.
- `run.sh`: top-level inference and packaging entry point.
- `Description.txt`: short method description.
- `CR__Geometry_Attribute_Temporal_Cascaded_Enhancement_for_Dual_Quality_Dynamic_Point_Clouds.pdf`: camera-ready method paper.
- `baselines/dqpc_b4/`: implementation, numbered pipeline scripts, evaluation helpers, and packaging tool.
- `external/`: referenced PU-Dense, GQE-Net, and UVG-CWI metric code used by the pipeline.

## Running

```bash
docker build -t gat-cascade .
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

The runner writes `outputs/dqpc_b4/submission_test.zip`.

## Validation

Local validation over 564 frames reports:

- Loose-threshold F-score: 0.8519
- Six-view projection SSIM: 0.9184

Runtime logs and optional diagnostic manifests are generated under `outputs/dqpc_b4/` during execution. The official package mode zips final RGB PLY files only.
