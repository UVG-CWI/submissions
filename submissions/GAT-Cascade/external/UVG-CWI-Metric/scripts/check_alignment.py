"""
Check alignment (scale and origin) between scan and ground truth point clouds.
Usage: python check_alignment.py <scan_path> <gt_path>
Supports .xyz, .ply, .pcd formats.
"""
import sys
import numpy as np
import os

def load_xyz(path):
    """Load XYZ file (space-separated, first 3 columns are x,y,z)"""
    return np.loadtxt(path, usecols=(0, 1, 2))

def load_ply(path):
    """Load PLY file using trimesh"""
    try:
        import trimesh
        mesh = trimesh.load(path, process=False)
        if hasattr(mesh, 'vertices'):
            return np.array(mesh.vertices)
        else:
            # Point cloud
            return np.array(mesh.vertices) if hasattr(mesh, 'vertices') else np.array(mesh)
    except Exception as e:
        print(f"Warning: Failed to load with trimesh: {e}")
        # Fallback: manual parsing
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
        
        # Find header end
        header_end = 0
        for i, line in enumerate(lines):
            if 'end_header' in line.lower():
                header_end = i + 1
                break
        
        # Parse data
        points = []
        for line in lines[header_end:]:
            parts = line.strip().split()
            if len(parts) >= 3:
                try:
                    points.append([float(parts[0]), float(parts[1]), float(parts[2])])
                except ValueError:
                    continue
        return np.array(points)

def load_points(path):
    """Load point cloud from various formats"""
    ext = os.path.splitext(path)[1].lower()
    if ext == '.xyz':
        return load_xyz(path)
    elif ext == '.ply':
        return load_ply(path)
    else:
        # Try as text file
        try:
            return load_xyz(path)
        except:
            raise ValueError(f"Unsupported file format: {ext}")

def check_alignment(scan_path, gt_path):
    """Check alignment between scan and ground truth"""
    print(f"\n{'='*60}")
    print(f"Alignment Check")
    print(f"{'='*60}")
    print(f"Scan: {scan_path}")
    print(f"GT:   {gt_path}")
    print(f"{'='*60}\n")
    
    # Load point clouds
    scan = load_points(scan_path)
    gt = load_points(gt_path)
    
    print(f"Point counts:")
    print(f"  Scan: {len(scan):,} points")
    print(f"  GT:   {len(gt):,} points")
    print()
    
    # Compute statistics
    for name, pts in [("Scan", scan), ("GT", gt)]:
        center = pts.mean(axis=0)
        mins = pts.min(axis=0)
        maxs = pts.max(axis=0)
        extents = maxs - mins
        
        print(f"{name} statistics:")
        print(f"  Centroid: [{center[0]:>10.4f}, {center[1]:>10.4f}, {center[2]:>10.4f}]")
        print(f"  Min:      [{mins[0]:>10.4f}, {mins[1]:>10.4f}, {mins[2]:>10.4f}]")
        print(f"  Max:      [{maxs[0]:>10.4f}, {maxs[1]:>10.4f}, {maxs[2]:>10.4f}]")
        print(f"  Extents:  [{extents[0]:>10.4f}, {extents[1]:>10.4f}, {extents[2]:>10.4f}]")
        print()
    
    # Compare alignment
    scan_center = scan.mean(axis=0)
    gt_center = gt.mean(axis=0)
    centroid_offset = np.linalg.norm(scan_center - gt_center)
    
    scan_extents = scan.max(axis=0) - scan.min(axis=0)
    gt_extents = gt.max(axis=0) - gt.min(axis=0)
    scale_ratios = scan_extents / (gt_extents + 1e-10)
    
    print(f"Alignment metrics:")
    print(f"  Centroid offset: {centroid_offset:.6f}")
    print(f"  Scale ratios (scan/gt): [{scale_ratios[0]:.6f}, {scale_ratios[1]:.6f}, {scale_ratios[2]:.6f}]")
    print()
    
    # Provide interpretation
    print(f"Interpretation:")
    if centroid_offset < 1.0:
        print(f"  ✓ Centroids are well-aligned (offset < 1.0)")
    else:
        print(f"  ✗ Centroids are NOT aligned (offset = {centroid_offset:.2f} >= 1.0)")
        print(f"    → Consider translating scan to match GT centroid")
    
    scale_close = np.all(np.abs(scale_ratios - 1.0) < 0.1)
    if scale_close:
        print(f"  ✓ Scales are consistent (ratios within 10% of 1.0)")
    else:
        print(f"  ✗ Scales differ significantly")
        print(f"    → Consider rescaling scan by factor: [{1/scale_ratios[0]:.4f}, {1/scale_ratios[1]:.4f}, {1/scale_ratios[2]:.4f}]")
    
    print(f"\n{'='*60}\n")
    
    if centroid_offset < 1.0 and scale_close:
        print("✓ Overall: Data appears well-aligned and ready for SAL training!")
    else:
        print("⚠ Warning: Alignment issues detected. Consider preprocessing before training.")
    
    return centroid_offset, scale_ratios

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python check_alignment.py <scan_path> <gt_path>")
        print("Example: python check_alignment.py dataset/real_object_scan/real_object_scan/bottle_shampoo.xyz dataset/real_object_GT/real_gt/bottle_shampoo.xyz")
        sys.exit(1)
    
    scan_path = sys.argv[1]
    gt_path = sys.argv[2]
    
    if not os.path.exists(scan_path):
        print(f"Error: Scan file not found: {scan_path}")
        sys.exit(1)
    
    if not os.path.exists(gt_path):
        print(f"Error: GT file not found: {gt_path}")
        sys.exit(1)
    
    check_alignment(scan_path, gt_path)
