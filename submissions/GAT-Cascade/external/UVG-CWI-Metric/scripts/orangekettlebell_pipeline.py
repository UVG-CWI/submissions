"""
Complete pipeline for SAL reconstruction and evaluation on UVG-CWI-DQPC dataset.

This script:
1. Prepares low-quality point clouds for SAL reconstruction
2. Trains SAL on selected frames
3. Reconstructs all frames
4. Evaluates reconstructed vs ground truth using vanilla metrics
"""
import os
import sys
import argparse
import trimesh
import numpy as np
from pathlib import Path
import subprocess
import time

def prepare_orangekettlebell_data(cg_dir, he_dir, sal_data_dir, test_frames=[0]):
    """
    Prepare OrangeKettlebell data for SAL.
    
    Args:
        cg_dir: directory with low-quality CG point clouds
        he_dir: directory with high-quality HE ground truth
        sal_data_dir: output directory for SAL
        test_frames: list of frame indices to prepare for testing (default: frame 0)
    """
    cg_dir = Path(cg_dir)
    he_dir = Path(he_dir)
    sal_data_dir = Path(sal_data_dir)
    
    points_dir = sal_data_dir / 'points' / 'OrangeKettlebell'
    points_iou_dir = sal_data_dir / 'points_iou' / 'OrangeKettlebell'
    points_dir.mkdir(parents=True, exist_ok=True)
    points_iou_dir.mkdir(parents=True, exist_ok=True)
    
    print("Preparing OrangeKettlebell data for SAL...")
    print(f"  CG (low-quality): {cg_dir}")
    print(f"  HE (ground truth): {he_dir}")
    print(f"  Output: {sal_data_dir}")
    print()
    
    # Get all CG and HE files
    cg_files = sorted(cg_dir.glob('*.ply'))
    he_files = sorted(he_dir.glob('*.ply'))
    
    if len(cg_files) == 0 or len(he_files) == 0:
        print("Error: No PLY files found!")
        return False
    
    print(f"Found {len(cg_files)} CG files and {len(he_files)} HE files")
    
    # Process test frames
    for frame_idx in test_frames:
        if frame_idx >= len(cg_files):
            print(f"Warning: Frame {frame_idx} not found, skipping")
            continue
            
        cg_file = cg_files[frame_idx]
        he_file = he_files[frame_idx]
        
        print(f"Processing frame {frame_idx:04d}: {cg_file.name}")
        
        # Load point clouds
        try:
            cg_mesh = trimesh.load(str(cg_file), process=False)
            he_mesh = trimesh.load(str(he_file), process=False)
            
            cg_points = np.array(cg_mesh.vertices)
            he_points = np.array(he_mesh.vertices)
            
            # Center-align (align CG to HE centroid)
            cg_center = cg_points.mean(axis=0)
            he_center = he_points.mean(axis=0)
            cg_aligned = cg_points - cg_center + he_center
            
            # Normalize to unit sphere for SAL
            all_points = np.vstack([cg_aligned, he_points])
            center = all_points.mean(axis=0)
            scale = np.abs(all_points - center).max()
            
            cg_normalized = (cg_aligned - center) / scale
            he_normalized = (he_points - center) / scale
            
            # Save normalized points
            cg_output = points_dir / f'frame_{frame_idx:04d}.xyz'
            he_output = points_iou_dir / f'frame_{frame_idx:04d}.xyz'
            
            np.savetxt(cg_output, cg_normalized, fmt='%.6f')
            np.savetxt(he_output, he_normalized, fmt='%.6f')
            
            print(f"  Saved: {cg_output.name} ({len(cg_normalized)} points)")
            print(f"         {he_output.name} ({len(he_normalized)} points)")
            print(f"  Offset: {np.linalg.norm(cg_center - he_center):.2f}, Scale: {scale:.2f}")
            
        except Exception as e:
            print(f"Error processing frame {frame_idx}: {e}")
            continue
    
    print()
    print(f"✓ Prepared {len(test_frames)} frames for SAL training/testing")
    return True

def train_sal_frame(frame_idx, sal_code_dir, epochs=500, resolution=128):
    """Train SAL on a single frame"""
    sal_code_dir = Path(sal_code_dir)
    
    print(f"\n{'='*60}")
    print(f"Training SAL on frame {frame_idx:04d}")
    print(f"{'='*60}\n")
    
    # Create config
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
    
    print(f"Config saved: {config_path}")
    print(f"Training for {epochs} epochs...")
    print()
    
    # Train
    cmd = [
        sys.executable,
        str(sal_code_dir / 'training/exp_runner.py'),
        '--batch_size', '1',
        '--nepoch', str(epochs),
        '--conf', str(config_path),
        '--workers', '1'
    ]
    
    result = subprocess.run(cmd, cwd=str(sal_code_dir))
    
    if result.returncode != 0:
        print(f"Training failed with code {result.returncode}")
        return False
    
    print(f"\n✓ Training completed for frame {frame_idx:04d}")
    return True

def evaluate_sal_frame(frame_idx, sal_code_dir, checkpoint, output_dir):
    """Evaluate SAL and extract mesh"""
    sal_code_dir = Path(sal_code_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Evaluating SAL frame {frame_idx:04d} at checkpoint {checkpoint}")
    print(f"{'='*60}\n")
    
    config_path = sal_code_dir / f'confs/orangekettlebell_frame_{frame_idx:04d}.conf'
    
    if not config_path.exists():
        print(f"Error: Config not found: {config_path}")
        return False
    
    # Evaluate
    cmd = [
        sys.executable,
        str(sal_code_dir / 'evaluate/evaluate.py'),
        '--conf', str(config_path),
        '--checkpoint', str(checkpoint),
        '--split', 'none'
    ]
    
    result = subprocess.run(cmd, cwd=str(sal_code_dir))
    
    if result.returncode != 0:
        print(f"Evaluation failed with code {result.returncode}")
        return False
    
    # Find generated mesh
    exp_dir = sal_code_dir / f'exps/OrangeKettlebell_frame_{frame_idx:04d}'
    eval_dir = exp_dir / 'evaluation'
    
    if eval_dir.exists():
        ply_files = list(eval_dir.glob('*.ply'))
        if ply_files:
            # Copy to output directory
            for ply_file in ply_files:
                import shutil
                output_file = output_dir / f'frame_{frame_idx:04d}_reconstructed.ply'
                shutil.copy(ply_file, output_file)
                print(f"✓ Reconstructed mesh saved: {output_file}")
                return True
    
    print(f"Warning: No PLY file found in {eval_dir}")
    return False

def run_vanilla_metrics(reconstructed_dir, gt_dir, output_csv):
    """Run vanilla metrics evaluation"""
    print(f"\n{'='*60}")
    print(f"Running vanilla metrics evaluation")
    print(f"{'='*60}\n")
    
    metrics_script = Path('third_party/SCUTSurface/metrics/vanilla_metric/eval_two_folder.py')
    
    if not metrics_script.exists():
        print(f"Error: Metrics script not found: {metrics_script}")
        return False
    
    cmd = [
        sys.executable,
        str(metrics_script),
        '--eval_type', 'syn_obj',
        '--in_dir', str(reconstructed_dir),
        '--gt_dir', str(gt_dir),
        '--samplepoints', '200000',
        '--out_csv', str(output_csv),
        '--num_worker', '4'
    ]
    
    result = subprocess.run(cmd)
    
    if result.returncode != 0:
        print(f"Metrics evaluation failed")
        return False
    
    print(f"✓ Metrics saved to: {output_csv}")
    return True

def main():
    parser = argparse.ArgumentParser(description='SAL reconstruction pipeline for UVG-CWI-DQPC')
    parser.add_argument('--cg_dir', type=str, 
                       default='dataset/UVG-CWI-DQPC/OrangeKettlebell/CG/15fps',
                       help='Directory with low-quality CG point clouds')
    parser.add_argument('--he_dir', type=str,
                       default='dataset/UVG-CWI-DQPC/OrangeKettlebell/HE/15fps',
                       help='Directory with high-quality HE ground truth')
    parser.add_argument('--test_frames', type=int, nargs='+', default=[0],
                       help='Frame indices to process (e.g., --test_frames 0 10 20)')
    parser.add_argument('--epochs', type=int, default=500,
                       help='Number of training epochs')
    parser.add_argument('--resolution', type=int, default=128,
                       help='Marching cubes resolution')
    parser.add_argument('--skip_train', action='store_true',
                       help='Skip training (use existing checkpoints)')
    parser.add_argument('--skip_eval', action='store_true',
                       help='Skip SAL evaluation')
    parser.add_argument('--skip_metrics', action='store_true',
                       help='Skip vanilla metrics evaluation')
    
    args = parser.parse_args()
    
    # Setup paths
    sal_data_dir = Path('third_party/SCUTSurface/reconstruction/SAL/data/uvg_kettlebell')
    sal_code_dir = Path('third_party/SCUTSurface/reconstruction/SAL/code')
    reconstructed_dir = Path('dataset/UVG-CWI-DQPC/OrangeKettlebell/Reconstructed')
    output_csv = Path('orangekettlebell_sal_metrics.csv')
    
    print(f"\n{'#'*60}")
    print("SAL Reconstruction Pipeline for OrangeKettlebell")
    print(f"{'#'*60}\n")
    print(f"Test frames: {args.test_frames}")
    print(f"Epochs: {args.epochs}")
    print(f"Resolution: {args.resolution}")
    print()
    
    # Step 1: Prepare data
    if not prepare_orangekettlebell_data(args.cg_dir, args.he_dir, sal_data_dir, args.test_frames):
        print("Data preparation failed!")
        return 1
    
    # Step 2: Train SAL on each frame
    if not args.skip_train:
        for frame_idx in args.test_frames:
            if not train_sal_frame(frame_idx, sal_code_dir, args.epochs, args.resolution):
                print(f"Training failed for frame {frame_idx}")
                continue
    
    # Step 3: Evaluate and extract meshes
    if not args.skip_eval:
        for frame_idx in args.test_frames:
            checkpoint = args.epochs
            if not evaluate_sal_frame(frame_idx, sal_code_dir, checkpoint, reconstructed_dir):
                print(f"Evaluation failed for frame {frame_idx}")
                continue
    
    # Step 4: Run vanilla metrics
    if not args.skip_metrics:
        if reconstructed_dir.exists() and any(reconstructed_dir.glob('*.ply')):
            run_vanilla_metrics(reconstructed_dir, args.he_dir, output_csv)
        else:
            print("No reconstructed files found, skipping metrics")
    
    print(f"\n{'#'*60}")
    print("Pipeline Complete!")
    print(f"{'#'*60}\n")
    print(f"Reconstructed files: {reconstructed_dir}")
    print(f"Metrics CSV: {output_csv}")

if __name__ == '__main__':
    main()
