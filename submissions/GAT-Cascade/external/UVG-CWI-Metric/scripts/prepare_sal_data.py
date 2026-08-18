"""
Prepare data for SAL reconstruction by aligning scan and GT pairs.
Copies and centers point clouds into SAL's expected data structure.
"""
import os
import sys
import numpy as np
import trimesh
from pathlib import Path

def load_points(path):
    """Load point cloud from .xyz or .ply"""
    ext = os.path.splitext(path)[1].lower()
    if ext == '.xyz':
        return np.loadtxt(path, usecols=(0, 1, 2))
    elif ext == '.ply':
        mesh = trimesh.load(path, process=False)
        if hasattr(mesh, 'vertices'):
            return np.array(mesh.vertices)
        return np.array(mesh)
    else:
        raise ValueError(f"Unsupported format: {ext}")

def save_xyz(points, path):
    """Save points to .xyz file"""
    np.savetxt(path, points, fmt='%.6f')

def center_align_pair(scan_path, gt_path, output_scan, output_gt):
    """Load, center-align to GT centroid, and save"""
    scan = load_points(scan_path)
    gt = load_points(gt_path)
    
    # Compute centroids
    scan_center = scan.mean(axis=0)
    gt_center = gt.mean(axis=0)
    
    # Translate scan to GT's centroid
    scan_aligned = scan - scan_center + gt_center
    
    # Also center GT to its own centroid (optional, for consistency)
    gt_centered = gt - gt_center + gt_center  # no-op, but can normalize
    
    # Save
    save_xyz(scan_aligned, output_scan)
    save_xyz(gt, output_gt)
    
    offset = np.linalg.norm(scan_center - gt_center)
    print(f"  Aligned: {os.path.basename(scan_path)} -> offset was {offset:.2f}")
    
    return scan_aligned, gt

def prepare_sal_dataset(scan_dir, gt_dir, sal_data_dir, object_names=None):
    """
    Prepare dataset for SAL reconstruction.
    
    Args:
        scan_dir: directory with scan .ply files (e.g., dataset/real_object_scan/real_object_scan)
        gt_dir: directory with GT .xyz files (e.g., dataset/real_object_GT/real_gt)
        sal_data_dir: output directory for SAL (will create points/ and points_iou/ subdirs)
        object_names: list of basenames to process (without extension), or None for all
    """
    scan_dir = Path(scan_dir)
    gt_dir = Path(gt_dir)
    sal_data_dir = Path(sal_data_dir)
    
    # Create SAL directory structure
    points_dir = sal_data_dir / 'points'
    points_iou_dir = sal_data_dir / 'points_iou'
    points_dir.mkdir(parents=True, exist_ok=True)
    points_iou_dir.mkdir(parents=True, exist_ok=True)
    
    # Find matching pairs
    scan_files = {f.stem.replace('_pcd', ''): f for f in scan_dir.glob('*_pcd.ply')}
    gt_files = {f.stem: f for f in gt_dir.glob('*.xyz')}
    
    common_names = set(scan_files.keys()) & set(gt_files.keys())
    
    if object_names:
        common_names = common_names & set(object_names)
    
    if not common_names:
        print("No matching pairs found!")
        return
    
    print(f"Found {len(common_names)} matching pairs")
    print(f"Output directory: {sal_data_dir}")
    print()
    
    for name in sorted(common_names):
        print(f"Processing: {name}")
        
        # Create object subdirectories
        obj_points_dir = points_dir / name
        obj_points_iou_dir = points_iou_dir / name
        obj_points_dir.mkdir(exist_ok=True)
        obj_points_iou_dir.mkdir(exist_ok=True)
        
        # Paths
        scan_path = scan_files[name]
        gt_path = gt_files[name]
        
        output_scan = obj_points_dir / f"{name}_scan.xyz"
        output_gt = obj_points_iou_dir / f"{name}_gt.xyz"
        
        # Align and save
        center_align_pair(scan_path, gt_path, output_scan, output_gt)
    
    print()
    print(f"✓ Prepared {len(common_names)} pairs for SAL")
    print(f"  Scans: {points_dir}")
    print(f"  Ground truth: {points_iou_dir}")
    print()
    print("Next steps:")
    print("  1. Navigate to SAL directory: cd third_party/SCUTSurface/reconstruction/SAL/code")
    print("  2. Create/edit config file pointing to your data")
    print("  3. Run training: python training/exp_runner.py --conf <your_config>.conf")

if __name__ == "__main__":
    # Default paths
    scan_dir = "dataset/real_object_scan/real_object_scan"
    gt_dir = "dataset/real_object_GT/real_gt"
    sal_data_dir = "third_party/SCUTSurface/reconstruction/SAL/data/real_objects"
    
    # Allow command-line override
    if len(sys.argv) > 1:
        scan_dir = sys.argv[1]
    if len(sys.argv) > 2:
        gt_dir = sys.argv[2]
    if len(sys.argv) > 3:
        sal_data_dir = sys.argv[3]
    
    prepare_sal_dataset(scan_dir, gt_dir, sal_data_dir)
