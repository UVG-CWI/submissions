"""
Simple SAL reconstruction for a single OrangeKettlebell frame.
This script trains SAL on one frame and generates the reconstructed mesh.

Usage: python train_sal_single_frame.py <frame_number> [--epochs 1000]
Example: python train_sal_single_frame.py 0 --epochs 500
"""
import sys
import os
import argparse
import numpy as np
import trimesh
from pathlib import Path

def prepare_frame_for_sal(cg_file, he_file, output_dir, frame_idx):
    """Prepare a single frame for SAL training"""
    output_dir = Path(output_dir)
    points_dir = output_dir / 'points' / 'OrangeKettlebell'
    points_iou_dir = output_dir / 'points_iou' / 'OrangeKettlebell'
    points_dir.mkdir(parents=True, exist_ok=True)
    points_iou_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Preparing frame {frame_idx}...")
    print(f"  CG: {cg_file}")
    print(f"  HE: {he_file}")
    
    # Load point clouds
    cg = trimesh.load(cg_file, process=False)
    he = trimesh.load(he_file, process=False)
    
    cg_pts = np.array(cg.vertices)
    he_pts = np.array(he.vertices)
    
    print(f"  CG points: {len(cg_pts)}")
    print(f"  HE points: {len(he_pts)}")
    
    # Center and normalize to unit sphere (SAL requirement)
    all_pts = np.vstack([cg_pts, he_pts])
    center = all_pts.mean(axis=0)
    scale = np.abs(all_pts - center).max()
    
    cg_norm = (cg_pts - center) / scale
    he_norm = (he_pts - center) / scale
    
    # Save
    cg_output = points_dir / f'frame_{frame_idx:04d}.xyz'
    he_output = points_iou_dir / f'frame_{frame_idx:04d}.xyz'
    
    np.savetxt(cg_output, cg_norm, fmt='%.8f')
    np.savetxt(he_output, he_norm, fmt='%.8f')
    
    print(f"  Saved normalized data:")
    print(f"    {cg_output}")
    print(f"    {he_output}")
    print(f"  Center: {center}")
    print(f"  Scale: {scale:.2f}")
    print()
    
    return str(cg_output), center, scale

def create_sal_config(frame_idx, sal_code_dir, epochs=1000, resolution=128):
    """Create SAL configuration file for a frame"""
    sal_code_dir = Path(sal_code_dir)
    
    config_content = f"""train{{
    plot_frequency = 50
    preprocess = True
    auto_decoder = False
    latent_size = 0
    expname = OrangeKettlebell_frame_{frame_idx:04d}
    dataset_path = ../data/uvg_kettlebell/points/OrangeKettlebell/frame_{frame_idx:04d}.xyz
    adjust_lr = False
    dataset = datasets.recon_dataset.ReconDataSet
    data_split = none

    learning_rate_schedule = [{{ "Type" : "Step",
                              "Initial" : 0.0005,
                               "Interval" : 500,
                                "Factor" : 0.5
                            }},
                            {{
                                "Type" : "Step",
                                "Initial" : 0.001,
                                "Interval" : 500,
                                "Factor" : 0.5
                            }}]
    network_class = model.network.SALNetwork
}}

plot{{
    resolution = {resolution}
    mc_value = 0.0
    is_uniform_grid = True
    verbose = False
    save_html = True
    save_ply = True
    overwrite = True
}}

network{{
    decode_mnfld_pnts = False
    encoder{{

    }}
    decoder
    {{
        dims = [ 512, 512, 512, 512, 512, 512, 512, 512 ]
        dropout = []
        dropout_prob =  0.2
        norm_layers = [0, 1, 2, 3, 4, 5, 6, 7]
        latent_in = []
        xyz_in_all = False
        activation = None

        latent_dropout = False
        weight_norm = True
    }}

    loss{{
        loss_type = model.loss.SALLoss
        properties{{
            manifold_pnts_weight = 0
            unsigned = True
        }}
    }}
}}
"""
    
    config_path = sal_code_dir / f'confs/orangekettlebell_frame_{frame_idx:04d}.conf'
    config_path.parent.mkdir(exist_ok=True)
    with open(config_path, 'w') as f:
        f.write(config_content)
    
    print(f"Config created: {config_path}")
    return config_path

def train_sal(config_path, sal_code_dir, epochs):
    """Train SAL"""
    import subprocess
    
    sal_code_dir = Path(sal_code_dir).absolute()
    config_path = Path(config_path).absolute()
    
    print(f"\n{'='*70}")
    print(f"Training SAL for {epochs} epochs (CPU mode)...")
    print(f"{'='*70}\n")
    
    cmd = [
        sys.executable,
        'training/exp_runner.py',
        '--batch_size', '1',
        '--nepoch', str(epochs),
        '--conf', str(config_path),
        '--workers', '1',
        '--gpu', '-1'  # Force CPU mode
    ]
    
    print(f"Working directory: {sal_code_dir}")
    print(f"Command: {' '.join(cmd)}")
    print()
    
    # Set environment to force CPU mode
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = ''  # Hide all CUDA devices
    
    result = subprocess.run(cmd, cwd=str(sal_code_dir), env=env)
    return result.returncode == 0

def evaluate_sal(config_path, sal_code_dir, checkpoint):
    """Evaluate SAL and extract mesh"""
    import subprocess
    
    sal_code_dir = Path(sal_code_dir).absolute()
    config_path = Path(config_path).absolute()
    
    print(f"\n{'='*70}")
    print(f"Evaluating SAL at checkpoint {checkpoint}...")
    print(f"{'='*70}\n")
    
    cmd = [
        sys.executable,
        'evaluate/evaluate.py',
        '--conf', str(config_path),
        '--checkpoint', str(checkpoint),
        '--split', 'none'
    ]
    
    result = subprocess.run(cmd, cwd=str(sal_code_dir))
    return result.returncode == 0

def rescale_mesh(mesh_file, output_ply, output_xyz, center, scale, num_samples=None):
    """
    Rescale mesh back to original coordinate system and save both mesh and point cloud.
    
    Args:
        mesh_file: Input mesh file (normalized)
        output_ply: Output PLY file for mesh
        output_xyz: Output XYZ file for point cloud
        center: Original center for denormalization
        scale: Original scale for denormalization
        num_samples: Number of points to sample (if None, use all vertices)
    """
    mesh = trimesh.load(mesh_file)
    
    # Get vertices (point cloud from mesh)
    vertices = np.array(mesh.vertices)
    
    # If mesh has faces, sample points from surface
    if hasattr(mesh, 'faces') and len(mesh.faces) > 0 and num_samples:
        from trimesh.sample import sample_surface
        vertices, _ = sample_surface(mesh, num_samples)
    
    # Reverse normalization: denormalize back to original scale
    vertices_rescaled = vertices * scale + center
    
    # Save rescaled mesh (PLY format with mesh structure)
    mesh_rescaled = trimesh.Trimesh(vertices=vertices_rescaled, faces=mesh.faces if hasattr(mesh, 'faces') else None)
    mesh_rescaled.export(output_ply)
    print(f"Rescaled mesh saved: {output_ply}")
    
    # Save as point cloud (XYZ format - plain text)
    np.savetxt(output_xyz, vertices_rescaled, fmt='%.6f')
    print(f"Reconstructed point cloud saved: {output_xyz}")
    
    # Also save as PLY point cloud (for compatibility with evaluation)
    pc_ply = trimesh.PointCloud(vertices=vertices_rescaled)
    output_pc_ply = str(output_ply).replace('.ply', '_pointcloud.ply')
    pc_ply.export(output_pc_ply)
    print(f"Reconstructed point cloud (PLY) saved: {output_pc_ply}")
    
    return output_ply, output_xyz, output_pc_ply

def main():
    parser = argparse.ArgumentParser(description='Train SAL on a single OrangeKettlebell frame')
    parser.add_argument('frame', type=int, help='Frame number to process (e.g., 0, 1, 2...)')
    parser.add_argument('--epochs', type=int, default=1000, help='Training epochs (default: 1000)')
    parser.add_argument('--resolution', type=int, default=128, help='Marching cubes resolution (default: 128)')
    parser.add_argument('--skip_train', action='store_true', help='Skip training (use existing checkpoint)')
    
    args = parser.parse_args()
    
    # Paths
    cg_dir = Path('dataset/UVG-CWI-DQPC/OrangeKettlebell/CG/15fps')
    he_dir = Path('dataset/UVG-CWI-DQPC/OrangeKettlebell/HE/15fps')
    sal_data_dir = Path('third_party/SCUTSurface/reconstruction/SAL/data/uvg_kettlebell')
    sal_code_dir = Path('third_party/SCUTSurface/reconstruction/SAL/code')
    reconstructed_dir = Path('dataset/UVG-CWI-DQPC/OrangeKettlebell/Reconstructed')
    reconstructed_dir.mkdir(parents=True, exist_ok=True)
    
    frame_idx = args.frame
    
    # Find files
    cg_files = sorted(cg_dir.glob('*.ply'))
    he_files = sorted(he_dir.glob('*.ply'))
    
    if frame_idx >= len(cg_files):
        print(f"Error: Frame {frame_idx} not found. Available: 0-{len(cg_files)-1}")
        return 1
    
    cg_file = cg_files[frame_idx]
    he_file = he_files[frame_idx]
    
    print(f"\n{'#'*70}")
    print(f"SAL Reconstruction - Frame {frame_idx}")
    print(f"{'#'*70}\n")
    
    # Step 1: Prepare data
    _, center, scale = prepare_frame_for_sal(cg_file, he_file, sal_data_dir, frame_idx)
    
    # Step 2: Create config
    config_path = create_sal_config(frame_idx, sal_code_dir, args.epochs, args.resolution)
    
    # Step 3: Train
    if not args.skip_train:
        if not train_sal(config_path, sal_code_dir, args.epochs):
            print("Training failed!")
            return 1
    
    # Step 4: Evaluate
    if not evaluate_sal(config_path, sal_code_dir, args.epochs):
        print("Evaluation failed!")
        return 1
    
    # Step 5: Find and rescale mesh to point cloud
    exp_dir = sal_code_dir / f'exps/OrangeKettlebell_frame_{frame_idx:04d}/evaluation'
    ply_files = list(exp_dir.glob('*.ply'))
    
    if ply_files:
        normalized_mesh = ply_files[0]
        
        # Output files in Reconstructed folder
        output_mesh = reconstructed_dir / f'frame_{frame_idx:04d}_mesh.ply'
        output_xyz = reconstructed_dir / f'frame_{frame_idx:04d}_reconstructed.xyz'
        
        # Rescale and save both mesh and point cloud
        mesh_file, xyz_file, pc_ply_file = rescale_mesh(
            normalized_mesh, output_mesh, output_xyz, center, scale, num_samples=500000
        )
        
        print(f"\n{'#'*70}")
        print(f"Success! Frame {frame_idx} reconstruction complete:")
        print(f"  Mesh:        {output_mesh}")
        print(f"  Point cloud: {xyz_file}")
        print(f"  Point cloud (PLY): {pc_ply_file}")
        print(f"{'#'*70}\n")
        print(f"To evaluate this reconstruction:")
        print(f"  python scripts/evaluate_orangekettlebell.py --mode reconstructed --max_frames {frame_idx+1}")
    else:
        print(f"Warning: No mesh found in {exp_dir}")
        return 1
    
    return 0

if __name__ == '__main__':
    sys.exit(main())
