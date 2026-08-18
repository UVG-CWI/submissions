# PointCloudEnhancement – Evaluation Metrics

This repository provides a complete Python implementation for evaluating the quality of 3D point-clouds, especially for point-cloud enhancement, reconstruction, and completion tasks.  
The metrics implemented here cover **accuracy**, **completeness**, **normal consistency**, **F-score**, and two types of **Chamfer distance**.

The core evaluation logic is implemented in  
`metrics.py`, which has been fully documented and explained in *Explanation.md*.

---

## 📦 Features

The toolkit computes the following metrics between a **predicted** point cloud and a **ground-truth** point cloud:

### **Geometry-based metrics**
- **Accuracy (CD_Acc)**  
  Mean nearest-neighbor distance from predicted → ground truth.
- **Completeness (CD_Comp)**  
  Mean nearest-neighbor distance from ground truth → predicted.
- **Symmetric Chamfer distances**  
  - `chamfer-L1` — L1 Chamfer distance (sum of mean distances in each direction)  
  - `chamfer-L2` — L2 Chamfer distance (sum of squared mean distances)  
  - `chamferL2_old` — legacy symmetric Chamfer = 0.5 × (accuracy + completeness)

### **Normal-based metrics**
- **Normal Accuracy (N_Acc)**  
  Cosine similarity between predicted normals and nearest neighbors in GT.
- **Normal Completeness (N_Comp)**  
  Cosine similarity from GT → predicted.
- **Normal Correctness (normals)**  
  Mean of the two normal terms (0.5 × N_Acc + 0.5 × N_Comp)

### **F-Score family (threshold-based metrics)**
Computed at configurable distance thresholds (default: `5, 10, 20` units):

For each τ:
- **Precision Pτ** = ratio of predicted points within τ of GT  
- **Recall Rτ** = ratio of GT points within τ of prediction  
- **Fτ = 2PR/(P+R)**

Example output keys:
P_5, R_5, F_5
P_10, R_10, F_10
P_20, R_20, F_20


---

## 🚀 Usage

### **Function call**
The core entry point is:

```python
from metrics import eval_pointcloud

results = eval_pointcloud(
    pre_mesh_ply="pred.ply",
    gt_mesh_ply="gt.ply",
    samplepoint=,
    eval_type=""
)


