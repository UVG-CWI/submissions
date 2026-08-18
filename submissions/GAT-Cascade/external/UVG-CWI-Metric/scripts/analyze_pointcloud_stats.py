"""
Analyze statistical properties of OrangeKettlebell point clouds.
Computes per-frame and average statistics for both HE (ground truth) and CG (low-quality) point clouds.

Statistics computed:
- Number of points
- Bounding box (min/max x, y, z)
- Centroid (center of mass)
- Average minimum distance between points (point density)
- Coordinate system extent
"""

import numpy as np
import trimesh
from pathlib import Path
from scipy.spatial import cKDTree
import pandas as pd
from tqdm import tqdm


def load_point_cloud(filepath):
    """Load point cloud from PLY file."""
    mesh = trimesh.load(filepath)
    if hasattr(mesh, 'vertices'):
        points = np.array(mesh.vertices)
    else:
        points = np.array(mesh)
    return points


def compute_min_distances(points, num_samples=5000, k=2):
    """
    Compute average minimum distance between points (point density metric).
    
    Args:
        points: Nx3 array of point coordinates
        num_samples: Number of points to sample for distance computation (for speed)
        k: Number of nearest neighbors (k=2 means 1st nearest neighbor, excluding self)
    
    Returns:
        mean_min_dist: Average distance to nearest neighbor
        std_min_dist: Standard deviation of distances
    """
    # Sample points if too many (for computational efficiency)
    if len(points) > num_samples:
        indices = np.random.choice(len(points), num_samples, replace=False)
        sampled_points = points[indices]
    else:
        sampled_points = points
    
    # Build KDTree and query nearest neighbors
    tree = cKDTree(points)
    distances, _ = tree.query(sampled_points, k=k)  # k=2 to get closest neighbor (excluding self)
    
    # distances[:, 1] is the distance to the nearest neighbor (excluding self at index 0)
    min_distances = distances[:, 1]
    
    return min_distances.mean(), min_distances.std()


def compute_statistics(points):
    """
    Compute comprehensive statistics for a point cloud.
    
    Returns:
        dict with statistics
    """
    stats = {}
    
    # Number of points
    stats['num_points'] = len(points)
    
    # Bounding box
    stats['min_x'] = points[:, 0].min()
    stats['max_x'] = points[:, 0].max()
    stats['min_y'] = points[:, 1].min()
    stats['max_y'] = points[:, 1].max()
    stats['min_z'] = points[:, 2].min()
    stats['max_z'] = points[:, 2].max()
    
    # Extent (range)
    stats['extent_x'] = stats['max_x'] - stats['min_x']
    stats['extent_y'] = stats['max_y'] - stats['min_y']
    stats['extent_z'] = stats['max_z'] - stats['min_z']
    
    # Centroid (center of mass)
    centroid = points.mean(axis=0)
    stats['centroid_x'] = centroid[0]
    stats['centroid_y'] = centroid[1]
    stats['centroid_z'] = centroid[2]
    
    # Average minimum distance (point density)
    mean_dist, std_dist = compute_min_distances(points)
    stats['avg_min_distance'] = mean_dist
    stats['std_min_distance'] = std_dist
    
    return stats


def analyze_orangekettlebell(max_frames=10):
    """
    Analyze OrangeKettlebell dataset statistics.
    """
    base_dir = Path(__file__).resolve().parents[1]
    cg_dir = base_dir / "dataset" / "UVG-CWI-DQPC" / "OrangeKettlebell" / "CG" / "15fps"
    he_dir = base_dir / "dataset" / "UVG-CWI-DQPC" / "OrangeKettlebell" / "HE" / "15fps"
    
    # Find files
    cg_files = sorted(cg_dir.glob("*.ply"))[:max_frames]
    he_files = sorted(he_dir.glob("*.ply"))[:max_frames]
    
    if not cg_files or not he_files:
        print("ERROR: No files found!")
        return
    
    print("="*80)
    print("POINT CLOUD STATISTICAL ANALYSIS")
    print("="*80)
    print(f"\nDataset: OrangeKettlebell")
    print(f"Frames analyzed: {max_frames}")
    print(f"CG files: {len(cg_files)}")
    print(f"HE files: {len(he_files)}")
    print()
    
    # Storage for per-frame statistics
    cg_stats_list = []
    he_stats_list = []
    
    # Process each frame
    print("Analyzing CG (low-quality) point clouds...")
    for i, cg_file in enumerate(tqdm(cg_files, desc="CG frames")):
        points = load_point_cloud(cg_file)
        stats = compute_statistics(points)
        stats['frame'] = i
        stats['filename'] = cg_file.name
        cg_stats_list.append(stats)
    
    print("\nAnalyzing HE (ground truth) point clouds...")
    for i, he_file in enumerate(tqdm(he_files, desc="HE frames")):
        points = load_point_cloud(he_file)
        stats = compute_statistics(points)
        stats['frame'] = i
        stats['filename'] = he_file.name
        he_stats_list.append(stats)
    
    # Convert to DataFrames
    cg_df = pd.DataFrame(cg_stats_list)
    he_df = pd.DataFrame(he_stats_list)
    
    # Save to CSV
    output_dir = base_dir / "results"
    output_dir.mkdir(exist_ok=True)
    
    cg_csv = output_dir / "orangekettlebell_cg_statistics.csv"
    he_csv = output_dir / "orangekettlebell_he_statistics.csv"
    
    cg_df.to_csv(cg_csv, index=False)
    he_df.to_csv(he_csv, index=False)
    
    print(f"\n✓ Saved CG statistics to: {cg_csv}")
    print(f"✓ Saved HE statistics to: {he_csv}")
    
    # Print summary
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    
    print("\n" + "-"*80)
    print("CG (Low-Quality) Point Clouds")
    print("-"*80)
    print_summary(cg_df)
    
    print("\n" + "-"*80)
    print("HE (Ground Truth) Point Clouds")
    print("-"*80)
    print_summary(he_df)
    
    print("\n" + "-"*80)
    print("COMPARISON (HE vs CG)")
    print("-"*80)
    print_comparison(he_df, cg_df)
    
    print("\n" + "="*80)


def print_summary(df):
    """Print summary statistics."""
    print(f"\n{'Metric':<30} {'Mean':<15} {'Std':<15} {'Min':<15} {'Max':<15}")
    print("-"*85)
    
    # Number of points
    print(f"{'Number of Points':<30} {df['num_points'].mean():>14.0f} "
          f"{df['num_points'].std():>14.0f} {df['num_points'].min():>14.0f} "
          f"{df['num_points'].max():>14.0f}")
    
    print("\nBounding Box (mm):")
    print(f"  {'X range [min, max]':<28} [{df['min_x'].mean():>8.2f}, {df['max_x'].mean():>8.2f}]  "
          f"extent: {df['extent_x'].mean():>8.2f}")
    print(f"  {'Y range [min, max]':<28} [{df['min_y'].mean():>8.2f}, {df['max_y'].mean():>8.2f}]  "
          f"extent: {df['extent_y'].mean():>8.2f}")
    print(f"  {'Z range [min, max]':<28} [{df['min_z'].mean():>8.2f}, {df['max_z'].mean():>8.2f}]  "
          f"extent: {df['extent_z'].mean():>8.2f}")
    
    print("\nCentroid (mm):")
    print(f"  {'X':<28} {df['centroid_x'].mean():>14.2f} {df['centroid_x'].std():>14.2f}")
    print(f"  {'Y':<28} {df['centroid_y'].mean():>14.2f} {df['centroid_y'].std():>14.2f}")
    print(f"  {'Z':<28} {df['centroid_z'].mean():>14.2f} {df['centroid_z'].std():>14.2f}")
    
    print("\nPoint Density:")
    print(f"  {'Avg Min Distance (mm)':<28} {df['avg_min_distance'].mean():>14.4f} "
          f"{df['avg_min_distance'].std():>14.4f}")
    print(f"  {'Std Min Distance (mm)':<28} {df['std_min_distance'].mean():>14.4f} "
          f"{df['std_min_distance'].std():>14.4f}")


def print_comparison(he_df, cg_df):
    """Print comparison between HE and CG."""
    print(f"\n{'Metric':<35} {'HE (GT)':<15} {'CG (LQ)':<15} {'Ratio (HE/CG)':<15}")
    print("-"*80)
    
    # Points
    he_points = he_df['num_points'].mean()
    cg_points = cg_df['num_points'].mean()
    print(f"{'Avg Number of Points':<35} {he_points:>14.0f} {cg_points:>14.0f} "
          f"{he_points/cg_points:>14.2f}x")
    
    # Point density
    he_density = he_df['avg_min_distance'].mean()
    cg_density = cg_df['avg_min_distance'].mean()
    print(f"{'Avg Min Distance (mm)':<35} {he_density:>14.4f} {cg_density:>14.4f} "
          f"{he_density/cg_density:>14.2f}x")
    
    # Extent comparison
    print("\nSpatial Extent Comparison:")
    he_extent_x = he_df['extent_x'].mean()
    cg_extent_x = cg_df['extent_x'].mean()
    print(f"  {'X extent (mm)':<33} {he_extent_x:>14.2f} {cg_extent_x:>14.2f} "
          f"{he_extent_x/cg_extent_x:>14.2f}x")
    
    he_extent_y = he_df['extent_y'].mean()
    cg_extent_y = cg_df['extent_y'].mean()
    print(f"  {'Y extent (mm)':<33} {he_extent_y:>14.2f} {cg_extent_y:>14.2f} "
          f"{he_extent_y/cg_extent_y:>14.2f}x")
    
    he_extent_z = he_df['extent_z'].mean()
    cg_extent_z = cg_df['extent_z'].mean()
    print(f"  {'Z extent (mm)':<33} {he_extent_z:>14.2f} {cg_extent_z:>14.2f} "
          f"{he_extent_z/cg_extent_z:>14.2f}x")
    
    # Centroid comparison (should be similar if aligned)
    print("\nCentroid Difference (mm):")
    diff_x = abs(he_df['centroid_x'].mean() - cg_df['centroid_x'].mean())
    diff_y = abs(he_df['centroid_y'].mean() - cg_df['centroid_y'].mean())
    diff_z = abs(he_df['centroid_z'].mean() - cg_df['centroid_z'].mean())
    print(f"  {'ΔX':<33} {diff_x:>14.2f}")
    print(f"  {'ΔY':<33} {diff_y:>14.2f}")
    print(f"  {'ΔZ':<33} {diff_z:>14.2f}")
    print(f"  {'Euclidean distance':<33} {np.sqrt(diff_x**2 + diff_y**2 + diff_z**2):>14.2f}")
    
    print("\nKey Observations:")
    if he_points > cg_points * 2:
        print(f"  • HE has {he_points/cg_points:.1f}x more points than CG (higher resolution)")
    if he_density < cg_density:
        print(f"  • HE has {cg_density/he_density:.1f}x denser point sampling than CG")
    if diff_x < 10 and diff_y < 10 and diff_z < 10:
        print(f"  • Point clouds are well-aligned (centroid difference < 10mm)")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Analyze point cloud statistics")
    parser.add_argument('--frames', type=int, default=10,
                       help='Number of frames to analyze (default: 10)')
    args = parser.parse_args()
    
    analyze_orangekettlebell(max_frames=args.frames)


if __name__ == "__main__":
    main()
