"""
Align CG point clouds to HE (ground truth) point clouds by translating centroids.
This fixes the misalignment issue before computing metrics.

Usage: python align_pointclouds.py --frames 10
"""
import numpy as np
import open3d as o3d
from pathlib import Path
import argparse


def align_pointcloud_pair(cg_path, he_path, output_path, verbose=True):
    """
    Align a single CG point cloud to its corresponding HE ground truth.
    
    Args:
        cg_path: Path to CG (low quality) PLY file
        he_path: Path to HE (ground truth) PLY file
        output_path: Path to save aligned CG PLY file
        verbose: Whether to print alignment info
    
    Returns:
        dict: Alignment metrics (centroid_offset_before, centroid_offset_after)
    """
    # Load point clouds
    cg_pcd = o3d.io.read_point_cloud(str(cg_path))
    he_pcd = o3d.io.read_point_cloud(str(he_path))
    
    cg_points = np.asarray(cg_pcd.points)
    he_points = np.asarray(he_pcd.points)
    
    # Compute centroids
    cg_centroid = cg_points.mean(axis=0)
    he_centroid = he_points.mean(axis=0)
    
    # Compute offset before alignment
    offset_before = np.linalg.norm(cg_centroid - he_centroid)
    
    # Translate CG to align with HE centroid
    translation = he_centroid - cg_centroid
    cg_points_aligned = cg_points + translation
    
    # Update point cloud
    cg_pcd.points = o3d.utility.Vector3dVector(cg_points_aligned)
    
    # Compute offset after alignment (should be ~0)
    cg_centroid_aligned = cg_points_aligned.mean(axis=0)
    offset_after = np.linalg.norm(cg_centroid_aligned - he_centroid)
    
    # Save aligned point cloud
    output_path.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_point_cloud(str(output_path), cg_pcd)
    
    if verbose:
        print(f"  Centroid offset before: {offset_before:.4f} mm")
        print(f"  Translation applied: [{translation[0]:.4f}, {translation[1]:.4f}, {translation[2]:.4f}]")
        print(f"  Centroid offset after: {offset_after:.6f} mm")
    
    return {
        'centroid_offset_before': offset_before,
        'centroid_offset_after': offset_after,
        'translation': translation
    }


def align_orangekettlebell_frames(max_frames=10):
    """Align CG frames to HE frames for OrangeKettlebell dataset."""
    base_dir = Path("dataset/UVG-CWI-DQPC/OrangeKettlebell")
    cg_dir = base_dir / "CG" / "15fps"
    he_dir = base_dir / "HE" / "15fps"
    output_dir = base_dir / "CG_aligned" / "15fps"
    
    # Find files
    cg_files = sorted(cg_dir.glob("*.ply"))[:max_frames]
    he_files = sorted(he_dir.glob("*.ply"))[:max_frames]
    
    if not cg_files or not he_files:
        print("ERROR: No files found!")
        return
    
    print("="*80)
    print("POINT CLOUD ALIGNMENT")
    print("="*80)
    print(f"\nDataset: OrangeKettlebell")
    print(f"Frames to align: {max_frames}")
    print(f"Source (CG): {cg_dir}")
    print(f"Target (HE): {he_dir}")
    print(f"Output: {output_dir}")
    print()
    
    results = []
    
    for i, (cg_file, he_file) in enumerate(zip(cg_files, he_files)):
        print(f"[{i+1}/{max_frames}] Aligning frame {i}...")
        print(f"  CG: {cg_file.name}")
        print(f"  HE: {he_file.name}")
        
        # Output file has same name as CG file
        output_file = output_dir / cg_file.name
        
        try:
            metrics = align_pointcloud_pair(cg_file, he_file, output_file, verbose=True)
            metrics['frame'] = i
            metrics['cg_file'] = cg_file.name
            metrics['he_file'] = he_file.name
            metrics['output_file'] = output_file.name
            results.append(metrics)
            
            print(f"  ✓ Saved to: {output_file.name}")
            print()
            
        except Exception as e:
            print(f"  ✗ ERROR: {e}")
            print()
            continue
    
    if results:
        print("="*80)
        print("ALIGNMENT SUMMARY")
        print("="*80)
        print()
        
        avg_offset_before = np.mean([r['centroid_offset_before'] for r in results])
        avg_offset_after = np.mean([r['centroid_offset_after'] for r in results])
        
        print(f"Average centroid offset:")
        print(f"  Before alignment: {avg_offset_before:.4f} mm")
        print(f"  After alignment:  {avg_offset_after:.6f} mm")
        print()
        print(f"✓ Aligned {len(results)} frames successfully!")
        print(f"✓ Output directory: {output_dir}")
        print("="*80)
    else:
        print("No frames were aligned!")


def main():
    parser = argparse.ArgumentParser(
        description="Align CG point clouds to HE ground truth by translating centroids"
    )
    parser.add_argument('--frames', type=int, default=10,
                       help='Number of frames to align (default: 10)')
    args = parser.parse_args()
    
    align_orangekettlebell_frames(max_frames=args.frames)


if __name__ == "__main__":
    main()
