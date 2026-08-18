"""
Poisson Surface Reconstruction Pipeline
Uses Open3D's traditional Poisson reconstruction (non-deep-learning method)

This script:
1. Loads low-quality point clouds (CG frames)
2. Estimates normals if not present
3. Applies Poisson surface reconstruction
4. Samples points from the reconstructed mesh
5. Saves reconstructed point clouds for evaluation
"""

import open3d as o3d
import numpy as np
import trimesh
import argparse
from pathlib import Path
from tqdm import tqdm


def load_point_cloud(ply_path):
    """Load point cloud from PLY file using trimesh."""
    mesh = trimesh.load(ply_path)
    if isinstance(mesh, trimesh.PointCloud):
        points = mesh.vertices
        colors = mesh.colors[:, :3] / 255.0 if mesh.colors is not None else None
        normals = mesh.vertex_normals if hasattr(mesh, 'vertex_normals') else None
    elif isinstance(mesh, trimesh.Trimesh):
        # Sample points from mesh
        points = mesh.vertices
        normals = mesh.vertex_normals
        colors = mesh.visual.vertex_colors[:, :3] / 255.0 if hasattr(mesh.visual, 'vertex_colors') else None
    else:
        raise ValueError(f"Unsupported mesh type: {type(mesh)}")
    
    return points, normals, colors


def estimate_normals(pcd, knn=30, radius=None):
    """
    Estimate normals for point cloud using KNN or radius search.
    
    Args:
        pcd: Open3D point cloud
        knn: Number of nearest neighbors for normal estimation
        radius: Search radius (if None, uses KNN)
    """
    if radius is not None:
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=knn)
        )
    else:
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamKNN(knn=knn)
        )
    
    # Orient normals consistently (towards camera/viewpoint)
    pcd.orient_normals_consistent_tangent_plane(k=knn)
    
    return pcd


def poisson_reconstruction(pcd, depth=8, scale=1.1, linear_fit=False):
    """
    Perform Screened Poisson Surface Reconstruction (SPSR).
    This is a traditional (non-deep-learning) method from:
    Kazhdan & Hoppe, "Screened Poisson Surface Reconstruction", 2013
    
    Args:
        pcd: Open3D point cloud with normals
        depth: Maximum depth of the octree (higher = more detail, slower)
        scale: Surface reconstruction scale (1.1 = 10% larger than bounding box)
        linear_fit: Use linear interpolation (faster but less accurate)
    
    Returns:
        mesh: Reconstructed triangle mesh
        densities: Vertex density values (can be used for filtering)
    """
    print(f"Running Screened Poisson Surface Reconstruction (depth={depth}, scale={scale})...")
    
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd,
        depth=depth,
        width=0,  # Auto-compute
        scale=scale,
        linear_fit=linear_fit
    )
    
    return mesh, densities


def ball_pivoting_reconstruction(pcd, radii=None):
    """
    Perform Ball Pivoting Algorithm (BPA) surface reconstruction.
    This is a traditional (non-deep-learning) triangulation-based method from:
    Bernardini et al., "The Ball-Pivoting Algorithm for Surface Reconstruction", 1999
    
    Args:
        pcd: Open3D point cloud with normals
        radii: List of ball radii for pivoting (if None, auto-compute from point cloud)
    
    Returns:
        mesh: Reconstructed triangle mesh
    """
    print(f"Running Ball Pivoting Algorithm (BPA) reconstruction...")
    
    if radii is None:
        # Auto-compute radii based on average nearest neighbor distance
        distances = pcd.compute_nearest_neighbor_distance()
        avg_dist = np.mean(distances)
        radii = [avg_dist, avg_dist * 2, avg_dist * 4]
    
    print(f"Using ball radii: {radii}")
    
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        pcd,
        o3d.utility.DoubleVector(radii)
    )
    
    return mesh


def filter_low_density_vertices(mesh, densities, quantile=0.01):
    """
    Remove low-density vertices (typically noise/outliers in reconstruction).
    
    Args:
        mesh: Triangle mesh
        densities: Vertex density array
        quantile: Remove vertices below this density quantile
    """
    densities = np.asarray(densities)
    density_threshold = np.quantile(densities, quantile)
    
    vertices_to_remove = densities < density_threshold
    mesh.remove_vertices_by_mask(vertices_to_remove)
    
    return mesh


def sample_points_from_mesh(mesh, num_points):
    """
    Sample points uniformly from reconstructed mesh surface.
    
    Args:
        mesh: Triangle mesh
        num_points: Number of points to sample
    
    Returns:
        pcd: Sampled point cloud
    """
    print(f"Sampling {num_points} points from mesh...")
    pcd = mesh.sample_points_uniformly(number_of_points=num_points)
    return pcd


def save_point_cloud(pcd, output_path):
    """Save point cloud to PLY file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_point_cloud(str(output_path), pcd)
    print(f"Saved reconstructed point cloud to {output_path}")


def reconstruct_single_frame(input_path, output_path, num_output_points=None, 
                            method='poisson', depth=8, knn=30, density_quantile=0.01,
                            bpa_radii=None):
    """
    Reconstruct a single point cloud frame using traditional methods.
    
    Args:
        input_path: Input PLY file path
        output_path: Output PLY file path
        num_output_points: Number of points to sample from reconstructed mesh 
                          (if None, use original point count)
        method: Reconstruction method ('poisson' or 'bpa')
        depth: Poisson octree depth (for Poisson method)
        knn: KNN for normal estimation
        density_quantile: Quantile threshold for density filtering (for Poisson method)
        bpa_radii: Ball radii for BPA (if None, auto-compute)
    """
    print(f"\nProcessing: {input_path}")
    
    # 1. Load point cloud
    points, normals, colors = load_point_cloud(input_path)
    print(f"Loaded {len(points)} points")
    
    if num_output_points is None:
        num_output_points = len(points)
    
    # 2. Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors)
    
    # 3. Estimate normals if not present
    if normals is None or len(normals) == 0:
        print(f"Estimating normals (knn={knn})...")
        pcd = estimate_normals(pcd, knn=knn)
    else:
        pcd.normals = o3d.utility.Vector3dVector(normals)
    
    # 4. Surface reconstruction
    if method.lower() == 'poisson':
        mesh, densities = poisson_reconstruction(pcd, depth=depth)
        print(f"Reconstructed mesh: {len(mesh.vertices)} vertices, {len(mesh.triangles)} triangles")
        
        # 5. Filter low-density vertices (remove noise)
        if density_quantile > 0:
            print(f"Filtering vertices with density < {density_quantile} quantile...")
            mesh = filter_low_density_vertices(mesh, densities, quantile=density_quantile)
            print(f"After filtering: {len(mesh.vertices)} vertices, {len(mesh.triangles)} triangles")
    
    elif method.lower() == 'bpa':
        mesh = ball_pivoting_reconstruction(pcd, radii=bpa_radii)
        print(f"Reconstructed mesh: {len(mesh.vertices)} vertices, {len(mesh.triangles)} triangles")
    
    else:
        raise ValueError(f"Unknown reconstruction method: {method}. Use 'poisson' or 'bpa'")
    
    # 6. Sample points from reconstructed mesh
    reconstructed_pcd = sample_points_from_mesh(mesh, num_output_points)
    
    # 7. Transfer colors if available (sample nearest neighbors from original)
    if colors is not None and len(colors) > 0:
        print("Transferring colors from original point cloud...")
        # Build KDTree for original points
        from scipy.spatial import cKDTree
        tree = cKDTree(points)
        reconstructed_points = np.asarray(reconstructed_pcd.points)
        
        # Find nearest neighbor in original cloud for each reconstructed point
        distances, indices = tree.query(reconstructed_points, k=1)
        reconstructed_colors = colors[indices]
        reconstructed_pcd.colors = o3d.utility.Vector3dVector(reconstructed_colors)
    
    # 8. Save reconstructed point cloud
    save_point_cloud(reconstructed_pcd, output_path)
    
    return reconstructed_pcd


def reconstruct_orangekettlebell_frames(frames_to_process=10, method='poisson',
                                       depth=8, knn=30, density_quantile=0.01,
                                       bpa_radii=None):
    """
    Reconstruct OrangeKettlebell CG frames using traditional reconstruction methods.
    
    Args:
        frames_to_process: Number of frames to process (default 10)
        method: Reconstruction method ('poisson' or 'bpa')
        depth: Poisson octree depth (8-10 recommended, for Poisson)
        knn: KNN for normal estimation
        density_quantile: Density filtering threshold (for Poisson)
        bpa_radii: Ball radii for BPA (if None, auto-compute)
    """
    base_dir = Path(__file__).resolve().parents[1]
    cg_dir = base_dir / "dataset" / "UVG-CWI-DQPC" / "OrangeKettlebell" / "CG" / "15fps"
    output_dir = base_dir / "dataset" / "UVG-CWI-DQPC" / "OrangeKettlebell" / "Reconstructed"
    
    # Find CG PLY files
    cg_files = sorted(cg_dir.glob("*.ply"))
    
    if not cg_files:
        print(f"ERROR: No CG files found in {cg_dir}")
        return
    
    print(f"Found {len(cg_files)} CG files")
    print(f"Processing first {frames_to_process} frames with {method.upper()} reconstruction")
    if method.lower() == 'poisson':
        print(f"Parameters: depth={depth}, knn={knn}, density_quantile={density_quantile}")
    else:
        print(f"Parameters: knn={knn}, radii={bpa_radii}")
    
    for i, cg_file in enumerate(tqdm(cg_files[:frames_to_process], desc="Poisson reconstruction")):
        frame_num = i  # 0-indexed
        output_file = output_dir / f"frame_{frame_num:04d}_pointcloud.ply"
        
        try:
            reconstruct_single_frame(
                input_path=cg_file,
                output_path=output_file,
                num_output_points=None,  # Match original point count
                method=method,
                depth=depth,
                knn=knn,
                density_quantile=density_quantile,
                bpa_radii=bpa_radii
            )
        except Exception as e:
            print(f"ERROR processing frame {frame_num}: {e}")
            continue
    
    
    print(f"\n{method.upper()} reconstruction complete! Results saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Traditional surface reconstruction methods (SPSR/BPA) - no deep learning"
    )
    parser.add_argument('--method', type=str, default='poisson', choices=['poisson', 'bpa'],
                       help='Reconstruction method: poisson (SPSR) or bpa (Ball Pivoting)')
    parser.add_argument('--frames', type=int, default=10, 
                       help='Number of frames to process (default: 10)')
    parser.add_argument('--depth', type=int, default=8, 
                       help='Poisson octree depth (8-10 recommended, higher=more detail)')
    parser.add_argument('--knn', type=int, default=30, 
                       help='KNN for normal estimation (default: 30)')
    parser.add_argument('--density_quantile', type=float, default=0.01, 
                       help='Density filtering quantile for Poisson (0.01 = remove 1%% lowest density)')
    parser.add_argument('--bpa_radii', type=float, nargs='+', default=None,
                       help='Ball radii for BPA (e.g., --bpa_radii 5.0 10.0 20.0)')
    parser.add_argument('--single_file', type=str, default=None,
                       help='Process single file instead of batch (provide input path)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output path for single file processing')
    
    args = parser.parse_args()
    
    if args.single_file:
        # Single file mode
        if not args.output:
            args.output = args.single_file.replace('.ply', '_reconstructed.ply')
        
        reconstruct_single_frame(
            input_path=args.single_file,
            output_path=args.output,
            method=args.method,
            depth=args.depth,
            knn=args.knn,
            density_quantile=args.density_quantile,
            bpa_radii=args.bpa_radii
        )
    else:
        # Batch mode for OrangeKettlebell
        reconstruct_orangekettlebell_frames(
            frames_to_process=args.frames,
            method=args.method,
            depth=args.depth,
            knn=args.knn,
            density_quantile=args.density_quantile,
            bpa_radii=args.bpa_radii
        )


if __name__ == "__main__":
    main()
