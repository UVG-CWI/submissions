
***

# Point Cloud Evaluation Metrics

A guide to common geometric metrics for evaluating the quality of a reconstructed point cloud against a ground-truth point cloud.

## 1. Setup and Notation

- Let **A** be the **ground-truth point cloud (GT)**.
- Let **B** be the **reconstructed point cloud (REC)**.

We denote by:
`d(x, Y) = min_{y ∈ Y} ‖x − y‖`
the Euclidean distance from a point `x` to the set `Y`.

## 2. Directional Metrics: A → B and B → A

### A → B (Completeness Direction)
- For each point `a ∈ A`, compute `d(a, B)`.
- This measures whether the ground-truth points are covered by the reconstruction.
- It highlights **missing parts** in the reconstruction (related to **completeness**).

### B → A (Accuracy Direction)
- For each point `b ∈ B`, compute `d(b, A)`.
- This measures whether reconstructed points have ground-truth support.
- It highlights **spurious points or noise** in the reconstruction (related to **precision**).

> **Note:** Both directions are necessary for a complete assessment. A→B captures missing data, while B→A captures false positives.

## 3. Precision, Recall, and F-Score (Threshold-Based)

Choose a distance threshold `τ` to judge if a point is correctly matched.

### Precision
`Precision(τ) = (1 / |B|) * Σ_{b ∈ B} 1{d(b, A) ≤ τ}`

- **Interpretation:** The fraction of reconstructed points that are within distance `τ` of some ground-truth point.
- Measures the **accuracy** of the reconstruction.

### Recall (Completeness)
`Recall(τ) = (1 / |A|) * Σ_{a ∈ A} 1{d(a, B) ≤ τ}`

- **Interpretation:** The fraction of ground-truth points that are within distance `τ` of some reconstructed point.
- Measures the **completeness** of the reconstruction.

### F-Score
`F(τ) = 2 * [Precision(τ) * Recall(τ)] / [Precision(τ) + Recall(τ)]`

- The harmonic mean of Precision and Recall, providing a single score that balances both.

## 4. Chamfer Distance

A common symmetric average distance metric:

`Chamfer(A, B) = (1 / |A|) * Σ_{a ∈ A} d(a, B)² + (1 / |B|) * Σ_{b ∈ B} d(b, A)²`

- This measures the **average squared nearest-neighbor distance** in both directions.
- It is robust to a few outliers and indicates the **overall alignment** of the point clouds.

## 5. Hausdorff Distance

The symmetric Hausdorff distance is defined as:

`H(A, B) = max( max_{a ∈ A} d(a, B), max_{b ∈ B} d(b, A) )`

- This captures the **worst-case nearest-neighbor error**.
- It is very sensitive to single large errors, outliers, or major missing regions.

## 6. Intuition and Practical Meaning

| Metric Pattern | Interpretation |
| :--- | :--- |
| **High Precision, Low Recall** | Reconstruction is accurate but incomplete (sparse). |
| **Low Precision, High Recall** | Reconstruction is complete but noisy/inaccurate. |
| **Low Chamfer & Low Hausdorff** | Good overall reconstruction with small average and worst-case errors. |
| **Low Chamfer, High Hausdorff** | Generally good reconstruction but with a few large errors/outliers. |

## 7. Threshold Selection Guidance

Choose the threshold `τ` relative to the point cloud's scale and density.

- **Heuristic:** Estimate the average nearest-neighbor spacing in the ground-truth cloud. Pick `τ` as a multiple (e.g., 1–3×) of that spacing.

## 8. Implementation Notes

- **Efficiency:** Use a KD-tree (e.g., `scipy.spatial.cKDTree`) for efficient nearest-neighbor queries.
- **Large Point Clouds:** For large datasets, consider subsampling or using GPU-accelerated implementations (e.g., for Chamfer Distance).
- **Color Metrics:** To do.
   

## 9. Example Numeric Walkthrough

**Given:**
- A → B distances: `[0.01, 0.02, 0.5, 0.015]`
- B → A distances: `[0.02, 0.01, 0.45, 0.1, 0.02]`
- Threshold `τ = 0.05`

**Calculations:**
- **Precision** = (Number of B→A distances ≤ 0.05) / |B| = `4 / 5 = 0.8`
- **Recall** = (Number of A→B distances ≤ 0.05) / |A| = `3 / 4 = 0.75`
- **F-Score** ≈ `2 * (0.8 * 0.75) / (0.8 + 0.75) ≈ 0.774`
- **Chamfer** ≈ `(mean of A→B squared distances) + (mean of B→A squared distances)` ≈ `0.128`
- **Hausdorff** = `max( max(A→B), max(B→A) ) = max(0.5, 0.45) = 0.5`

---

## Appendix: Quick-Start Checklist

1.  **Verify Units:** Confirm the units of our point clouds (e.g., meters? I guess millimeters).
2.  **Choose Threshold `τ`:** Select `τ` based on the estimated point spacing in our HE ground-truth data (still doesn't know how).
3.  **Compute Distances:** Use a KD-tree to compute nearest-neighbor distances for both A→B and B→A.
4.  **Calculate Metrics:**
    - Compute Precision, Recall, and F-Score using the threshold `τ`.
    - Compute Chamfer Distance (mean of squared distances).
    - Compute Hausdorff Distance (max of nearest-neighbor distances).


***



