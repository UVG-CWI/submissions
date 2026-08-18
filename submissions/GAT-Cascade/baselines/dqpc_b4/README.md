# DQPC B4 Baseline: PU-Dense + Gaussian/DA-KNN Recoloring + GQE-Net

This folder records the selected baseline structure for GC2026 UVG-CWI-DQPC:

```text
CG_15 point cloud
  -> PU-Dense geometry enhancement
  -> Gaussian/DA-KNN recoloring from CG/RGB-D colors to enhanced geometry
  -> GQE-Net color refinement
  -> per-frame PLY(x, y, z, red, green, blue) + runtime log + manifest
```

## Downloaded Code

The reusable upstream repositories are kept under `external/`:

| Module | Purpose | Local path | Upstream |
|---|---|---|---|
| PU-Dense | Geometry upsampling/enhancement | `external/PointCloudUpsampling` | `https://github.com/aniqueakhtar/PointCloudUpsampling.git` |
| GQE-Net | Color attribute enhancement | `external/GQE-Net` | `https://github.com/xjr998/GQE-Net.git` |
| UVG-CWI Metric | Optional official-style geometry validation | `external/UVG-CWI-Metric` | `https://github.com/UVG-CWI/Metric` |

Current downloaded commits:

```text
PU-Dense: 64f4347
GQE-Net: 5044852
UVG-CWI Metric: main branch zip downloaded on 2026-05-24, no .git metadata
```

The DUGAE paper declares `https://github.com/yuanhui0325/DUGAE`, but `git clone` currently returns `Repository not found`. DA-KNN/Gaussian recoloring should therefore be implemented locally instead of treated as a downloadable dependency.

`external/UVG-CWI-Metric` is kept as a validation reference only. It is not used
by the training, inference, recoloring, temporal smoothing, or PLY packaging
steps. Its Python implementation imports `scipy` and `trimesh`, so run it with a
Python environment that includes those packages.

## Selected Main Structure

Use **PU-Dense as the main engineering structure** and integrate GQE-Net as a post-geometry color branch.

Reasoning:

- The competition requires enhanced geometry and RGB PLY output. PU-Dense already owns the geometry generation step, sparse tensor network, paired PLY loading path, and large-cloud KD-tree inference pattern.
- `external/PointCloudUpsampling/data.py::Dataset` already supports `input .ply + GT_folder .ply`, which matches `CG_15 -> HE_15` better than GQE-Net's h5 patch workflow.
- `external/PointCloudUpsampling/model/Network.py::MyNet` is the right network entry to adapt for DQPC geometry. It is a Minkowski sparse U-Net that predicts high-LoD occupancy and prunes generated coordinates.
- GQE-Net should not be the top-level scaffold because it assumes fixed-size 1024/2048 point patches, h5 training files, three separate Y/U/V models, and static single-frame color enhancement.

So the baseline should modify the PU-Dense path first:

```text
external/PointCloudUpsampling/
  data.py                  -> replace synthetic/downsample dataset with DQPC paired sequence dataset
  train.py                 -> add DQPC train/valid arguments and sequence-aware logging
  eval_*.py or new script  -> add DQPC inference without GT, output PLY + runtime
  model/Network.py::MyNet  -> keep as initial geometry backbone; only adjust input quantization, output ratio, pruning, and ME version compatibility
```

Then adapt GQE-Net as a second-stage module:

```text
external/GQE-Net/
  model_GQE_Net_final.py::GAPCN  -> keep as color refinement backbone
  data.py / main_mix.py          -> replace h5/G-PCC patch generator with DQPC enhanced-PLY patch sampler
```

## Required DQPC Adaptations

### 1. Data Adapter

Create a DQPC dataset layer that scans paired frames:

```text
DQPC_ROOT/
  train/<sequence>/consumer-grade_capture_system/CG_aligned/15fps/*.ply
  train/<sequence>/high-end_capture_system/HE/15fps/*.ply
  valid/<sequence>/consumer-grade_capture_system/CG_aligned/15fps/*.ply
  valid/<sequence>/high-end_capture_system/HE/15fps/*.ply
  test/<sequence>/consumer-grade_capture_system/CG_aligned/15fps/*.ply
```

The adapter should return:

```text
input_coords_int:  [N, 3] quantized CG coordinates
input_rgb:         [N, 3] CG RGB
target_coords_int: [M, 3] quantized HE coordinates, train/valid only
target_rgb:        [M, 3] HE RGB, train/valid only
metadata:          sequence id, frame id, original scale, file paths
```

PU-Dense only needs occupancy features initially, so use `ones` as sparse features for geometry training. Keep RGB in metadata for recoloring and GQE-Net.

### 2. Geometry Network

Keep `MyNet` as the first geometry baseline.

Main changes:

- Train from real `CG_15 -> HE_15`, not synthetic downsampled ShapeNet.
- Quantize coordinates using the dataset voxel size, preferably 1 mm to match the DQPC high-end voxelization note in `summary.md`.
- Use overlapping KD-tree or spatial blocks for million-point frames.
- During training, use `coords_T` from HE as occupancy targets.
- During inference/test, run without `coords_T`; keep top-k or score-thresholded generated coordinates per block, then merge duplicate voxels.
- Write enhanced geometry as PLY before recoloring.

### 3. Gaussian/DA-KNN Recoloring

After PU-Dense produces enhanced coordinates, transfer attributes from the original CG frame or RGB-D projection:

```text
source: CG points with RGB
target: PU-Dense enhanced coordinates
output: enhanced coordinates with initial RGB
```

Default formula for target point `p`:

```text
w_i = exp(-||p - q_i||^2 / (2 * sigma^2))
rgb(p) = sum_i w_i * rgb(q_i) / sum_i w_i
```

Recommended defaults:

```text
k = 3 or 8
sigma = median distance to the kth neighbor in the current frame/block
fallback = nearest-neighbor color if all weights are near zero
```

`recolor_gaussian.py` in this folder provides the initial standalone implementation.

DA-KNN extension:

- Use geometry distance plus color consistency/normal consistency in the weights.
- Prefer neighbors from the same connected local surface where normal estimates are reliable.
- For newly generated points near holes, optionally combine CG colors with raw RGB-D projection colors.

### 4. GQE-Net Color Branch

Keep `GAPCN` as the color backbone but replace the original data path.

Original input per patch:

```text
[x, y, z, one_yuv_channel] -> enhanced one_yuv_channel
```

DQPC training input:

```text
PU-Dense enhanced xyz + recolored Y/U/V -> HE-matched Y/U/V target
```

Needed changes:

- Generate patches from enhanced DQPC frames, not h5 files.
- Match each enhanced point to HE colors by nearest-neighbor or Gaussian correspondence for supervision.
- Train three channel-specific models first, matching the original GQE-Net design.
- Keep overlap-allowed patch inference and average duplicated predictions.
- Add sequence/frame logging and write final RGB PLY.

### 5. Temporal Wrapper

For the first runnable B4 baseline, keep the model single-frame and add only light postprocess temporal smoothing:

```text
RGB_t = alpha * RGB_t + (1 - alpha) * mapped RGB_{t-1/t+1}
```

After B4 is stable, add STQE-style recoloring-based motion compensation and temporal attention around GQE-Net.

## Minimal Build Order

1. B0 identity: copy `CG_15` to output and validate PLY/manifest/runtimes.
2. B2 geometry: adapt PU-Dense to `CG_15 -> HE_15`; output geometry only.
3. B2+ recoloring: run `recolor_gaussian.py` from CG RGB to PU-Dense geometry.
4. B3 color: adapt GQE-Net patch training on DQPC recolored enhanced frames.
5. B4 cascade: run geometry, recoloring, and GQE-Net in one sequence-level script.

## Runnable Scripts Added

This folder now contains the DQPC adaptation code for steps 2 and 3:

| File | Role |
|---|---|
| `dqpc_data.py` | DQPC PLY I/O, `CG_15 -> HE_15` pair discovery, coordinate quantization, random training crops, sparse batch collation |
| `train_pudense_dqpc.py` | PU-Dense geometry fine-tuning on paired DQPC frames |
| `infer_pudense_dqpc.py` | PU-Dense geometry inference on CG frames, with KD-tree block partition and runtime JSONL logging |
| `recolor_sequence.py` | Batch Gaussian KNN recoloring from CG RGB to PU-Dense enhanced geometry |
| `scripts/01_train_pudense_dqpc.sh` | One-command training wrapper |
| `scripts/02_infer_geometry_and_recolor.sh` | One-command geometry inference + recoloring wrapper |
| `scripts/00_check_env_and_data.sh` | Environment and dataset-layout checks |
| `scripts/03_train_gqenet_dqpc.sh` | GQE-Net DQPC Y/U/V channel training |
| `scripts/04_infer_gqenet_dqpc.sh` | GQE-Net Y/U/V inference and patch fusion |
| `scripts/05_identity_baseline.sh` | Identity baseline, copies CG frames to output layout |
| `scripts/06_evaluate_basic.sh` | Basic local evaluation: NN geometry error, F-score, scale diagnostics, and YUV PSNR |
| `scripts/07_make_submission.sh` | PLY-only submission zip by default; optional full internal package |
| `scripts/08_temporal_smooth.sh` | Lightweight temporal color smoothing across adjacent frames |
| `scripts/09_da_knn_recolor.sh` | DA-KNN recoloring with distance, normal, and color-consistency weights |
| `scripts/10_evaluate_external.sh` | Optional wrapper for pc_error and PCQM binaries; PCQM is supplementary |
| `scripts/11_evaluate_projection.sh` | Six-view projection SSIM and optional LPIPS proxy |
| `scripts/12_summarize_validation.sh` | Combine UVG Metric/basic/external/projection/runtime summaries into one validation board |
| `scripts/13_evaluate_uvg_metric.sh` | Preferred official-style geometry validation through `external/UVG-CWI-Metric` |
| `scripts/14_calibrate_gqenet_blend.sh` | Calibrate validation Y/U/V blend weights and apply them without rerunning GQE-Net |
| `scripts/15_luminance_lut.sh` | Fit an HE-supervised monotonic Y LUT and apply it without changing XYZ |
| `scripts/16_transfer_color_donor.sh` | Transfer colors from a strong V3/donor result to the current geometry |

The `scripts/` directory is the numbered shell launcher layer. The evaluation
Python implementations stay in `baselines/dqpc_b4/` and are invoked by the
numbered `.sh` wrappers.

## Performance Profiles and V3 Compatibility

The default color path keeps the V3 behavior that produced the established YUV
PSNR baseline:

- `TRAIN_PROFILE=v3_compatible`
- absolute Y/U/V supervision
- `COORD_SCALE=1.0`
- `CENTER_MODE=v3_stride`
- exact Gaussian/DA-KNN global scale statistics

The following options improve throughput without changing the color formula:

- KNN queries are processed with `QUERY_CHUNK_SIZE` instead of materializing all
  target-neighbor tensors.
- GQE-Net shares patch indices and normalized XYZ across Y/U/V inference.
- GQE-Net uses broadcast distance weights instead of allocating repeated
  130/640-channel tensors.
- `DISTANCE_UNIT=mm` keeps physical thresholds consistent for both millimeter
  and meter coordinate files.

Quality-affecting experiments remain opt-in:

- `TRAIN_PROFILE=v3_compatible_fast` groups patches from the same frame to reuse
  PLY/KD-tree caches while retaining absolute V3 labels.
- `CENTER_MODE=spatial_voxel` replaces point-order patch centers with spatial
  coverage centers.
- `COLOR_SCALE_MODE=sampled` accelerates DA-KNN scale estimation; the default
  `exact` mode preserves the full-data median.

For checkpoint selection by held-out color MSE/PSNR, set
`VAL_ENHANCED_ROOT`, `VAL_ENHANCED_GLOB`, and `VAL_HE_ROOT` during GQE-Net
training. The best checkpoint is written as `best.pth`.


## Recommended Run Order

Run from the repository root and set the dataset/Python environment first:

```bash
cd /home/fansonglin/xieliang/ACMMM-DQPC
export PYTHON_BIN=/path/to/conda/env/bin/python
export DQPC_ROOT=/path/to/UVG-CWI-DQPC
export SPLIT=valid
```

Recommended sequence:

1. Check environment and dataset layout.

```bash
bash baselines/dqpc_b4/scripts/00_check_env_and_data.sh
```

2. Generate the identity baseline.

```bash
bash baselines/dqpc_b4/scripts/05_identity_baseline.sh
```

3. Run a PU-Dense debug training job first.

```bash
export SPLIT=train
export STEPS=20
export SAVE_EVERY=20
export OUT_DIR=outputs/dqpc_b4/pudense_ckpts_debug
bash baselines/dqpc_b4/scripts/01_train_pudense_dqpc.sh
```

4. Run full PU-Dense training.

```bash
export SPLIT=train
export STEPS=2000
export SAVE_EVERY=500
export OUT_DIR=outputs/dqpc_b4/pudense_ckpts
bash baselines/dqpc_b4/scripts/01_train_pudense_dqpc.sh
```

5. Run PU-Dense geometry inference plus Gaussian recoloring.

```bash
export SPLIT=valid
export PUDENSE_CKPT=outputs/dqpc_b4/pudense_ckpts/iter2000.pth
bash baselines/dqpc_b4/scripts/02_infer_geometry_and_recolor.sh
```

6. Evaluate the B2+ output.

```bash
export PRED_ROOT=outputs/dqpc_b4/valid_colored
bash baselines/dqpc_b4/scripts/06_evaluate_basic.sh
```

7. Train GQE-Net Y/U/V models.

```bash
export SPLIT=train
export ENHANCED_ROOT=outputs/dqpc_b4/train_colored
export HE_ROOT=$DQPC_ROOT/train
export TRAIN_PROFILE=v3_compatible
for CHANNEL in 0 1 2; do
  export CHANNEL
  bash baselines/dqpc_b4/scripts/03_train_gqenet_dqpc.sh
done
```

8. Run raw GQE-Net inference. `PREDICTION_MODE=auto` reads the checkpoint
metadata; legacy/V3/upstream checkpoints without metadata are treated as
absolute-output models.

```bash
export SPLIT=valid
export INPUT_ROOT=outputs/dqpc_b4/valid_colored
export OUT_ROOT=outputs/dqpc_b4/valid_gqenet_raw
export MODEL_Y=outputs/dqpc_b4/gqenet_ckpts/y/model_19.pth
export MODEL_U=outputs/dqpc_b4/gqenet_ckpts/u/model_19.pth
export MODEL_V=outputs/dqpc_b4/gqenet_ckpts/v/model_19.pth
export PREDICTION_MODE=auto
bash baselines/dqpc_b4/scripts/04_infer_gqenet_dqpc.sh
```

9. Calibrate conservative Y/U/V blending on validation and write the final B4
output. The search includes blend `0`, so each selected channel is no worse
than the pre-GQE color baseline under the same validation correspondence and
YUV PSNR calculation. The tool then verifies the written RGB round-trip and
falls back to the full pre-GQE color baseline if any channel regresses.

```bash
export DQPC_ROOT=/path/to/UVG-CWI-DQPC
export SPLIT=valid
export BASE_ROOT=outputs/dqpc_b4/valid_colored
export CANDIDATE_ROOT=outputs/dqpc_b4/valid_gqenet_raw
export OUT_ROOT=outputs/dqpc_b4/valid_gqenet
bash baselines/dqpc_b4/scripts/14_calibrate_gqenet_blend.sh
```

10. Evaluate the full B4 output.

```bash
export PRED_ROOT=outputs/dqpc_b4/valid_gqenet
bash baselines/dqpc_b4/scripts/06_evaluate_basic.sh
```

11. Optionally smooth colors temporally.

```bash
export INPUT_ROOT=outputs/dqpc_b4/valid_gqenet
export OUT_ROOT=outputs/dqpc_b4/valid_temporal
bash baselines/dqpc_b4/scripts/08_temporal_smooth.sh
```

12. Package a PLY-only submission archive.

```bash
export INPUT_ROOT=outputs/dqpc_b4/valid_temporal
bash baselines/dqpc_b4/scripts/07_make_submission.sh
```

The default follows the current organizer guidance and zips only the final PLY
files. For internal bookkeeping only, add metadata with:

```bash
export PACKAGE_MODE=full
export VALIDATE_RGB=1
bash baselines/dqpc_b4/scripts/07_make_submission.sh
```

13. Run the bundled UVG-CWI Metric for the preferred geometry summary.

```bash
export PRED_ROOT=outputs/dqpc_b4/valid_temporal
bash baselines/dqpc_b4/scripts/13_evaluate_uvg_metric.sh
```

This writes `outputs/dqpc_b4/eval/<split>_uvg_metric_eval_summary.json`.
When present, `12_summarize_validation.sh` uses these `CD_Acc`, `CD_Comp`,
`chamfer-L1`, `chamfer-L2`, and `F_5/F_10/F_20/F_30` values as the primary
geometry metrics.

14. Optionally run external metrics if binaries are installed.

```bash
export PRED_ROOT=outputs/dqpc_b4/valid_temporal
export PC_ERROR_BIN=/path/to/pc_error_d
bash baselines/dqpc_b4/scripts/10_evaluate_external.sh
```

The script auto-detects the bundled PCQM binary when present:

```text
external/PCQM/build/PCQM
```

The wrapper runs PCQM with `--fastquit`, copies required PCQM resource files to
`outputs/dqpc_b4/eval/pcqm_work` by default, and writes
`outputs/dqpc_b4/eval/<split>_external_eval_summary.json` when PCQM values are
parsed.

15. Run six-view projection metrics.

```bash
export PRED_ROOT=outputs/dqpc_b4/valid_temporal
bash baselines/dqpc_b4/scripts/11_evaluate_projection.sh
```

Optional LPIPS requires the `lpips` Python package and weights:

```bash
export COMPUTE_LPIPS=1
bash baselines/dqpc_b4/scripts/11_evaluate_projection.sh
```

16. Summarize validation metrics into one board.

```bash
export RUN_NAME=valid_temporal
bash baselines/dqpc_b4/scripts/12_summarize_validation.sh
```

DA-KNN recoloring can replace Step 5's Gaussian recoloring output after PU-Dense geometry is generated:

```bash
export DQPC_ROOT=/path/to/UVG-CWI-DQPC
export SPLIT=valid
export TARGET_ROOT=outputs/dqpc_b4/valid_geometry
export OUT_ROOT=outputs/dqpc_b4/valid_daknn_colored
bash baselines/dqpc_b4/scripts/09_da_knn_recolor.sh
```

### Expected Dataset Layout

The default shell scripts assume:

```text
$DQPC_ROOT/
  train/<sequence>/consumer-grade_capture_system/CG_aligned/15fps/*.ply
  train/<sequence>/high-end_capture_system/HE/15fps/*.ply
  valid/<sequence>/consumer-grade_capture_system/CG_aligned/15fps/*.ply
  valid/<sequence>/high-end_capture_system/HE/15fps/*.ply
  test/<sequence>/consumer-grade_capture_system/CG_aligned/15fps/*.ply
```

The scripts also auto-detect compact legacy layouts such as `CG_aligned/15fps`,
`CGv2_15`, and `CG/15fps`. If the real dataset layout differs, override
`CG_GLOB`, `HE_GLOB`, or `INPUT_ROOT` in the shell command.

### Step 0: Check Environment and Dataset

```bash
export PYTHON_BIN=/path/to/conda/env/bin/python
export DQPC_ROOT=/path/to/UVG-CWI-DQPC
export SPLIT=train

bash baselines/dqpc_b4/scripts/00_check_env_and_data.sh
```

Outputs:

```text
outputs/dqpc_b4/checks/env.json
outputs/dqpc_b4/checks/dataset_train.json
```

### Step 2: Train/Fine-Tune PU-Dense Geometry

Run from the repository root inside the PU-Dense-compatible environment:

```bash
export DQPC_ROOT=/path/to/UVG-CWI-DQPC
export SPLIT=train
export STEPS=2000
export OUT_DIR=outputs/dqpc_b4/pudense_ckpts

bash baselines/dqpc_b4/scripts/01_train_pudense_dqpc.sh
```

Optional overrides:

```bash
export INIT_CKPT=/path/to/pudense_pretrained_or_previous_ckpt.pth
export VOXEL_SIZE=auto
export CROP_SIZE=256
export MAX_CG_POINTS=70000
export MAX_HE_POINTS=280000
export BATCH_SIZE=1
export SAVE_EVERY=500
```

Outputs:

```text
outputs/dqpc_b4/pudense_ckpts/
  train_log.jsonl
  iter500.pth
  iter1000.pth
  ...
```

### Step 3: PU-Dense Geometry Inference + Gaussian Recoloring

Run after training has produced a checkpoint:

```bash
export DQPC_ROOT=/path/to/UVG-CWI-DQPC
export SPLIT=valid
export PUDENSE_CKPT=outputs/dqpc_b4/pudense_ckpts/iter2000.pth
export SKIP_EXISTING=0

bash baselines/dqpc_b4/scripts/02_infer_geometry_and_recolor.sh
```

Optional overrides:

```bash
export INPUT_ROOT=$DQPC_ROOT/valid
export CG_GLOB="$INPUT_ROOT/*/consumer-grade_capture_system/CG_aligned/15fps/*.ply"
export GEOMETRY_OUT_ROOT=outputs/dqpc_b4/valid_geometry
export COLORED_OUT_ROOT=outputs/dqpc_b4/valid_colored
export RUNTIME_DIR=outputs/dqpc_b4/runtime
export UP_RATIO=4
export TARGET_POINT_RATIO=
export SCORE_THRESHOLD=
export MAX_OUTPUT_POINT_RATIO=6
export BLOCK_HALO=32
export VOXEL_SIZE=auto
export INCLUDE_INPUT=1
export K=8
export SIGMA=
export COPY_DISTANCE=0.5
export LIMIT=0
```

Outputs:

```text
outputs/dqpc_b4/valid_geometry/<relative DQPC input path>/*.ply
outputs/dqpc_b4/valid_colored/<relative DQPC input path>/*.ply
outputs/dqpc_b4/runtime/valid_geometry.jsonl
outputs/dqpc_b4/runtime/valid_recolor.jsonl
```

For a quick smoke test on only one frame:

```bash
export DQPC_ROOT=/path/to/UVG-CWI-DQPC
export SPLIT=valid
export PUDENSE_CKPT=outputs/dqpc_b4/pudense_ckpts/iter2000.pth
export LIMIT=1

bash baselines/dqpc_b4/scripts/02_infer_geometry_and_recolor.sh
```

### Step 4: Train GQE-Net Y/U/V Models

Run after Step 3 has produced colored enhanced PLY for the training split.

```bash
export PYTHON_BIN=/path/to/conda/env/bin/python
export DQPC_ROOT=/path/to/UVG-CWI-DQPC
export SPLIT=train
export ENHANCED_ROOT=outputs/dqpc_b4/train_colored
export HE_ROOT=$DQPC_ROOT/train
export EPOCHS=20
export SAMPLES_PER_EPOCH=10000
export TRAIN_PROFILE=v3_compatible

for CHANNEL in 0 1 2; do
  export CHANNEL
  bash baselines/dqpc_b4/scripts/03_train_gqenet_dqpc.sh
done
```

Outputs:

```text
outputs/dqpc_b4/gqenet_ckpts/y/model_*.pth
outputs/dqpc_b4/gqenet_ckpts/u/model_*.pth
outputs/dqpc_b4/gqenet_ckpts/v/model_*.pth
```

`TRAIN_PROFILE=v3_compatible` is the default and reproduces the V3 objective:
absolute Y/U/V regression with ordinary MSE over all nearest HE targets.
`TRAIN_PROFILE=robust_residual` enables residual prediction and down-weights
HE correspondences beyond 20 coordinate units. Treat that profile as a
separate experiment, not as a drop-in V3 replacement.

### Step 5: Infer GQE-Net Color Refinement

```bash
export PYTHON_BIN=/path/to/conda/env/bin/python
export SPLIT=valid
export INPUT_ROOT=outputs/dqpc_b4/valid_colored
export OUT_ROOT=outputs/dqpc_b4/valid_gqenet_raw
export MODEL_Y=outputs/dqpc_b4/gqenet_ckpts/y/model_19.pth
export MODEL_U=outputs/dqpc_b4/gqenet_ckpts/u/model_19.pth
export MODEL_V=outputs/dqpc_b4/gqenet_ckpts/v/model_19.pth
export PREDICTION_MODE=auto

bash baselines/dqpc_b4/scripts/04_infer_gqenet_dqpc.sh
```

Output:

```text
outputs/dqpc_b4/valid_gqenet_raw/<relative DQPC input path>/*.ply
outputs/dqpc_b4/runtime/valid_gqenet.jsonl
```

`PREDICTION_MODE=auto` is required for mixed checkpoint provenance:

- checkpoints written by current training code carry `prediction_mode`;
- old DQPC/V3 checkpoints without this field are treated as absolute YUV;
- upstream `external/GQE-Net/pths/...` checkpoints are also absolute YUV.

Do not force `RESIDUAL=1` for a checkpoint that was trained to predict absolute
YUV. That mode mismatch was the main code path capable of causing the observed
large Y/U/V PSNR drop. Residual inference also now leaves points without patch
coverage unchanged instead of adding their input color twice.

### Step 5.1: Calibrate GQE-Net Color Blending

```bash
export DQPC_ROOT=/path/to/UVG-CWI-DQPC
export SPLIT=valid
export BASE_ROOT=outputs/dqpc_b4/valid_colored
export CANDIDATE_ROOT=outputs/dqpc_b4/valid_gqenet_raw
export OUT_ROOT=outputs/dqpc_b4/valid_gqenet

bash baselines/dqpc_b4/scripts/14_calibrate_gqenet_blend.sh
```

The selected validation weights are saved to:

```text
outputs/dqpc_b4/eval/valid_gqenet_blend.json
```

To reject a validation result that does not reach the required color floor:

```bash
export MIN_PSNR_Y=17.3
export MIN_PSNR_U=29.5
export MIN_PSNR_V=29.5
export MIN_GAIN_Y=0.05
export REQUIRE_QUALITY_GATE=1
bash baselines/dqpc_b4/scripts/14_calibrate_gqenet_blend.sh
```

The gate is evaluated after the actual `YUV -> RGB uint8 -> YUV` write path.
When it fails, the diagnostic JSON is retained but the final PLY tree is not
written. This is a validation-set acceptance gate, not a mathematical
guarantee for hidden test data.

`MIN_GAIN_Y=0.05` also rejects a LUT/GQE candidate whose measured validation
gain is too small to justify test-set risk. In that case the saved JSON reports
`selection_decision: base_fallback` and the final output is copied from the
base through zero blend weights.

Calibration also stores a `color_drift_guard` profile from valid frames. During
test application, any frame whose mean or 99th-percentile Y/U/V change, or RGB
clipping increase, exceeds that valid profile falls back to the base for that
frame. Regenerate the blend JSON after this change; older JSON files do not
contain the drift profile.

Apply the same validation-derived weights to test outputs:

```bash
export MODE=apply
export SPLIT=test
export BASE_ROOT=outputs/dqpc_b4/test_colored
export CANDIDATE_ROOT=outputs/dqpc_b4/test_gqenet_raw
export CONFIG_IN=outputs/dqpc_b4/eval/valid_gqenet_blend.json
export OUT_ROOT=outputs/dqpc_b4/test_gqenet

bash baselines/dqpc_b4/scripts/14_calibrate_gqenet_blend.sh
```

### Step 5.2: Fit and Apply the Y-LUT

Fit the monotonic luminance LUT using train outputs and train HE references:

```bash
export MODE=fit
export SPLIT=train
export SOURCE_ROOT=outputs/dqpc_b4/train_colored
export MODEL_PATH=outputs/dqpc_b4/y_lut/train_y_lut.json
bash baselines/dqpc_b4/scripts/15_luminance_lut.sh
```

Apply it to validation data, then use script 14 with
`CANDIDATE_ROOT=outputs/dqpc_b4/valid_y_lut_raw` to select a safe Y/U/V blend:

```bash
export MODE=apply
export SPLIT=valid
export SOURCE_ROOT=outputs/dqpc_b4/valid_colored
export OUT_ROOT=outputs/dqpc_b4/valid_y_lut_raw
bash baselines/dqpc_b4/scripts/15_luminance_lut.sh
```

The LUT corrects only Y before RGB serialization. Integer conversion and gamut
clipping can still perturb U/V slightly, so script 14 remains the required
validation guard. A global LUT can correct systematic luminance bias, but it
cannot recover missing local texture or guarantee a fixed Y PSNR.

### Step 5.3: Preserve V3 Color on Current Geometry

If the V3 result already reaches approximately 30 dB on U/V, transfer its RGB
to the current geometry before optional Y-LUT/GQE refinement:

```bash
export DONOR_ROOT=/path/to/V3/valid_result
export TARGET_ROOT=outputs/dqpc_b4/valid_geometry
export OUT_ROOT=outputs/dqpc_b4/valid_v3_color_current_geometry
bash baselines/dqpc_b4/scripts/16_transfer_color_donor.sh
```

The donor and target trees must have the same relative frame layout. This step
copies color by nearest geometry while preserving the target XYZ and point
count. U/V near 30 dB is only preserved when the donor itself has that
performance and the geometry correspondence remains accurate.

### Step 6: Identity Baseline

```bash
export PYTHON_BIN=/path/to/conda/env/bin/python
export DQPC_ROOT=/path/to/UVG-CWI-DQPC
export SPLIT=valid

bash baselines/dqpc_b4/scripts/05_identity_baseline.sh
```

### Step 7: Basic Local Evaluation

Evaluate any output tree that preserves the relative DQPC PLY layout:

```bash
export PYTHON_BIN=/path/to/conda/env/bin/python
export DQPC_ROOT=/path/to/UVG-CWI-DQPC
export SPLIT=valid
export PRED_ROOT=outputs/dqpc_b4/valid_colored

bash baselines/dqpc_b4/scripts/06_evaluate_basic.sh
```

This writes nearest-neighbor geometry diagnostics, YUV PSNR, and F-score
threshold sweeps. Override thresholds in the PLY coordinate unit when the
official value is known:

```bash
export FSCORE_THRESHOLDS=5,10,20,30
bash baselines/dqpc_b4/scripts/06_evaluate_basic.sh
```

For final GQE-Net output:

```bash
export PRED_ROOT=outputs/dqpc_b4/valid_gqenet
bash baselines/dqpc_b4/scripts/06_evaluate_basic.sh
```

### Step 7.1: UVG-CWI Metric Geometry Evaluation

Use this as the main geometry metric output before comparing methods for the
competition:

```bash
export PYTHON_BIN=/path/to/conda/env/bin/python
export DQPC_ROOT=/path/to/UVG-CWI-DQPC
export SPLIT=valid
export PRED_ROOT=outputs/dqpc_b4/valid_gqenet

bash baselines/dqpc_b4/scripts/13_evaluate_uvg_metric.sh
```

PCQM remains available through `scripts/10_evaluate_external.sh` as an optional
perceptual supplement; it should not replace the UVG-CWI geometry metrics.

## Environment Notes

PU-Dense expects an older stack:

```text
Python 3.7/3.8
PyTorch 1.6/1.7
MinkowskiEngine 0.4
```

The current base Python in this workspace is 3.13 and is not sufficient for the
full pipeline: `scipy`, `plyfile`, and `MinkowskiEngine` are missing, and CUDA is
not available from that environment. Training, inference, and UVG-CWI Metric
evaluation should be done in a dedicated conda environment with the needed
packages. The bundled UVG-CWI Metric specifically imports `scipy` and `trimesh`.

For the standalone recoloring script only:

```bash
pip install -r baselines/dqpc_b4/requirements-recolor.txt
python baselines/dqpc_b4/recolor_gaussian.py \
  --source path/to/CG_frame.ply \
  --target path/to/pudense_geometry.ply \
  --output path/to/recolored_frame.ply \
  --k 8
```
