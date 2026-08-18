# Scripts Directory

This directory contains various Python scripts for point cloud processing, evaluation, and reconstruction tasks on the OrangeKettlebell dataset from UVG-CWI-DQPC.

## Table of Contents

- [Alignment and Preprocessing](#alignment-and-preprocessing)
- [Statistical Analysis](#statistical-analysis)
- [Reconstruction Methods](#reconstruction-methods)
- [Evaluation and Metrics](#evaluation-and-metrics)
- [Training Scripts](#training-scripts)
- [Utility Scripts](#utility-scripts)

---

## Alignment and Preprocessing

### `align_pointclouds.py`
Aligns CG (low-quality) point clouds to HE (ground truth) point clouds by translating centroids.

**Usage:**
```bash
python scripts/align_pointclouds.py --frames 10
```

**Options:**
- `--frames N`: Number of frames to align (default: 10)

**Output:** Aligned point clouds saved to `dataset/UVG-CWI-DQPC/OrangeKettlebell/CG_aligned/15fps/`

**Note:** This performs simple centroid-based alignment. May not be suitable for all use cases.

---

### `check_alignment.py`
Checks alignment between scan and ground truth point clouds by comparing centroids, scales, and extents.

**Usage:**
```bash
python scripts/check_alignment.py <scan_path> <gt_path>
```

**Example:**
```bash
python scripts/check_alignment.py \
    dataset/UVG-CWI-DQPC/OrangeKettlebell/CG/15fps/OrangeKettlebell_UVG-CWI-DQPC_CG_15_0_169_0000.ply \
    dataset/UVG-CWI-DQPC/OrangeKettlebell/HE/15fps/OrangeKettlebell_UVG-CWI-DQPC_HE_15_0_169_0000.ply
```

**Output:** Console report showing centroid offset, scale ratios, and alignment recommendations

**Supports:** `.xyz`, `.ply`, `.pcd` formats

---

## Statistical Analysis

### `analyze_pointcloud_stats.py`
Computes statistical information about point clouds including density, bounding boxes, centroids, and point spacing.

**Usage:**
```bash
python scripts/analyze_pointcloud_stats.py --frames 10
```

**Options:**
- `--frames N`: Number of frames to analyze (default: 10)

**Output:** 
- `results/orangekettlebell_cg_statistics.csv` - Statistics for CG (low-quality) point clouds
- `results/orangekettlebell_he_statistics.csv` - Statistics for HE (ground truth) point clouds

**Metrics computed:**
- Number of points
- Bounding box (min/max X, Y, Z coordinates)
- Extents (width, height, depth)
- Centroid position
- Average minimum distance between points (density metric)
- Standard deviation of minimum distances

---

## Reconstruction Methods

### `poisson_reconstruction.py`
Implements traditional surface reconstruction methods: Screened Poisson Surface Reconstruction (SPSR) and Ball Pivoting Algorithm (BPA).

**Usage:**
```bash
# Run SPSR reconstruction
python scripts/poisson_reconstruction.py --method spsr --frames 10

# Run BPA reconstruction
python scripts/poisson_reconstruction.py --method bpa --frames 10
```

**Options:**
- `--method {spsr,bpa}`: Reconstruction method to use (default: spsr)
- `--frames N`: Number of frames to process (default: 10)

**Output:** Reconstructed meshes saved to `dataset/UVG-CWI-DQPC/OrangeKettlebell/Reconstructed/`

**SPSR Parameters:**
- Octree depth: 9
- Density quantile for filtering: 0.01
- Normal estimation: KNN=30

**BPA Parameters:**
- Radii computed based on average nearest neighbor distances
- Uses 3 different radii for multi-scale reconstruction

---

## Evaluation and Metrics

### `run_vanilla_metrics.py`
Runs comprehensive point cloud evaluation using the SCUTSurface vanilla_metric toolkit. Computes Chamfer distances, precision/recall/F-scores, and normal consistency metrics.

**Usage:**
```bash
python scripts/run_vanilla_metrics.py --frames 10
```

**Options:**
- `--frames N`: Number of frames to evaluate (default: 10)

**Output:** `results/orangekettlebell_vanilla_metrics.csv`

**Metrics computed:**
- **Chamfer-L1**: Bidirectional L1 Chamfer distance (mm)
- **Chamfer-L2**: Bidirectional L2 Chamfer distance (mm²)
- **CD_Acc**: Accuracy - average distance from predicted to GT (mm)
- **CD_Comp**: Completeness - average distance from GT to predicted (mm)
- **Precision @ τ**: Percentage of predicted points within threshold τ of GT
- **Recall @ τ**: Percentage of GT points within threshold τ of predicted
- **F-score @ τ**: Harmonic mean of precision and recall at threshold τ
- **N_Acc**: Normal accuracy (cosine similarity)
- **N_Comp**: Normal completeness
- **normals**: Overall normal correctness

**Current thresholds:** [5, 10, 20] mm

**Note:** Uses Open3D to estimate normals (KNN=20) if not present in PLY files

---

### `evaluate_orangekettlebell.py`
Baseline evaluation script using custom Chamfer distance implementation. Compares CG (low-quality) against HE (ground truth).

**Usage:**
```bash
python scripts/evaluate_orangekettlebell.py --max_frames 10
```

**Options:**
- `--max_frames N`: Number of frames to evaluate (default: 10)

**Output:** `results/orangekettlebell_baseline_metrics.csv`

**Thresholds:** [10, 20, 30, 50] mm

---

### `traditional_reconstruction_comparison.py`
Evaluates reconstructed point clouds (e.g., from SPSR) against ground truth and compares with baseline.

**Usage:**
```bash
python scripts/traditional_reconstruction_comparison.py --frames 10
```

**Output:** `results/orangekettlebell_reconstructed_metrics.csv`

---

### `traditional_methods_summary.py`
Displays a summary comparison between baseline and reconstructed metrics. Shows improvement percentages.

**Usage:**
```bash
python scripts/traditional_methods_summary.py
```

**Input files:**
- `results/orangekettlebell_traditional_methods_baseline.csv`
- `results/orangekettlebell_traditional_methods_reconstructed.csv`

**Output:** Formatted console output showing side-by-side comparison

---

## Training Scripts

### `train_sal_object.py`
Training script for SAL (Surface-Aligned Latent) model on object-level point cloud completion.

**Usage:**
```bash
python scripts/train_sal_object.py
```

**Note:** Requires SAL model implementation in `third_party/SCUTSurface/reconstruction/SAL/`

---

### `train_sal_single_frame.py`
Training script for SAL model on single frame point cloud completion.

**Usage:**
```bash
python scripts/train_sal_single_frame.py
```

---

### `prepare_sal_data.py`
Prepares data in the format required for SAL training.

**Usage:**
```bash
python scripts/prepare_sal_data.py
```

---

## Utility Scripts

### `save_separate_csv_results.py`
Splits combined evaluation results into separate CSV files for baseline and reconstructed metrics.

**Usage:**
```bash
python scripts/save_separate_csv_results.py
```

**Output:**
- `results/orangekettlebell_traditional_methods_baseline.csv`
- `results/orangekettlebell_traditional_methods_reconstructed.csv`

---

### `orangekettlebell_pipeline.py`
End-to-end pipeline script that runs multiple steps in sequence.

**Usage:**
```bash
python scripts/orangekettlebell_pipeline.py
```

**Steps typically include:**
1. Baseline evaluation
2. Reconstruction
3. Reconstructed evaluation
4. Summary generation

---

## Common Workflows

### 1. Analyze Point Cloud Statistics
```bash
# Check alignment between CG and HE
python scripts/check_alignment.py \
    dataset/UVG-CWI-DQPC/OrangeKettlebell/CG/15fps/OrangeKettlebell_UVG-CWI-DQPC_CG_15_0_169_0000.ply \
    dataset/UVG-CWI-DQPC/OrangeKettlebell/HE/15fps/OrangeKettlebell_UVG-CWI-DQPC_HE_15_0_169_0000.ply

# Compute detailed statistics
python scripts/analyze_pointcloud_stats.py --frames 10
```

### 2. Run Vanilla Metrics Evaluation (No Alignment)
```bash
# Evaluate with small thresholds [5, 10, 20]mm
python scripts/run_vanilla_metrics.py --frames 10

# View results
cat results/orangekettlebell_vanilla_metrics.csv
```

### 3. Traditional Reconstruction and Evaluation
```bash
# Step 1: Baseline evaluation
python scripts/evaluate_orangekettlebell.py --max_frames 10

# Step 2: Run SPSR reconstruction
python scripts/poisson_reconstruction.py --method spsr --frames 10

# Step 3: Evaluate reconstructed point clouds
python scripts/traditional_reconstruction_comparison.py --frames 10

# Step 4: View comparison summary
python scripts/traditional_methods_summary.py
```

### 4. Alignment Workflow (if needed)
```bash
# Step 1: Check alignment
python scripts/check_alignment.py <cg_file> <he_file>

# Step 2: Align point clouds (if necessary)
python scripts/align_pointclouds.py --frames 10

# Step 3: Re-check alignment
python scripts/check_alignment.py \
    dataset/UVG-CWI-DQPC/OrangeKettlebell/CG_aligned/15fps/<aligned_file> \
    dataset/UVG-CWI-DQPC/OrangeKettlebell/HE/15fps/<he_file>
```

---

## Dependencies

All scripts require the following Python packages:
- `numpy`
- `scipy`
- `pandas`
- `open3d` (v0.19.0+)
- `trimesh`
- `plyfile`

Install with:
```bash
pip install numpy scipy pandas open3d trimesh plyfile
```

---

## Notes

### Point Cloud Formats
- Most scripts support `.ply` format
- `check_alignment.py` also supports `.xyz` and `.pcd`

### Dataset Structure
Expected directory structure:
```
dataset/UVG-CWI-DQPC/OrangeKettlebell/
├── CG/15fps/           # Low-quality (compressed) point clouds
├── HE/15fps/           # High-quality (ground truth) point clouds
├── CG_aligned/15fps/   # Aligned CG point clouds (generated)
└── Reconstructed/      # Reconstructed meshes/point clouds (generated)
```

### Output Directory
All results are saved to `results/` directory:
```
results/
├── orangekettlebell_baseline_metrics.csv
├── orangekettlebell_reconstructed_metrics.csv
├── orangekettlebell_vanilla_metrics.csv
├── orangekettlebell_cg_statistics.csv
├── orangekettlebell_he_statistics.csv
└── ...
```

### Known Issues

1. **Alignment**: The `align_pointclouds.py` script performs simple centroid-based alignment, which may not be appropriate for all scenarios. Consider using ICP or other robust alignment methods if needed.

2. **F-score bug fix**: The vanilla_metrics evaluation had a bug where F-scores were computed on scalar means instead of per-point distances. This has been fixed in `metrics.py` (lines 90-97).

3. **scipy compatibility**: The `distance_p2p()` function in `metrics.py` was updated to use `workers=-1` instead of the deprecated `n_jobs=8` parameter for scipy 1.6+.

---

## For More Information

See the main repository README and documentation:
- [../README.md](../README.md)


