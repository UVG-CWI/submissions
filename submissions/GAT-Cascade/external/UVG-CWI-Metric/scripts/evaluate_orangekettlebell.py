"""
Simple evaluation workflow for OrangeKettlebell dataset.
Step 1: Evaluate baseline (low-quality CG vs ground truth HE)
Step 2: After SAL reconstruction, evaluate reconstructed vs ground truth
"""
import os
import sys
import csv
import numpy as np
from pathlib import Path
from scipy.spatial import cKDTree
import trimesh

def compute_chamfer_distance(pc1, pc2, sample_points=None):
    """Compute Chamfer distance between two point clouds"""
    if sample_points:
        if len(pc1) > sample_points:
            idx = np.random.choice(len(pc1), sample_points, replace=False)
            pc1 = pc1[idx]
        if len(pc2) > sample_points:
            idx = np.random.choice(len(pc2), sample_points, replace=False)
            pc2 = pc2[idx]
    
    # Build KD-trees
    tree1 = cKDTree(pc1)
    tree2 = cKDTree(pc2)
    
    # Compute distances
    dist_1_to_2, _ = tree1.query(pc2, k=1)
    dist_2_to_1, _ = tree2.query(pc1, k=1)
    
    # Chamfer distance
    chamfer_dist = (dist_1_to_2.mean() + dist_2_to_1.mean()) / 2
    
    # Accuracy and Completeness
    accuracy = dist_2_to_1.mean()
    completeness = dist_1_to_2.mean()
    
    # Hausdorff distance
    hausdorff_dist = max(dist_1_to_2.max(), dist_2_to_1.max())
    
    return {
        'chamfer_distance': chamfer_dist,
        'accuracy': accuracy,
        'completeness': completeness,
        'hausdorff_distance': hausdorff_dist
    }

def compute_precision_recall_fscore(pc1, pc2, threshold=0.01):
    """Compute precision, recall, and F-score at a given threshold"""
    tree1 = cKDTree(pc1)
    tree2 = cKDTree(pc2)
    
    dist_1_to_2, _ = tree1.query(pc2, k=1)
    dist_2_to_1, _ = tree2.query(pc1, k=1)
    
    precision = (dist_2_to_1 < threshold).sum() / len(dist_2_to_1)
    recall = (dist_1_to_2 < threshold).sum() / len(dist_1_to_2)
    
    if precision + recall > 0:
        fscore = 2 * precision * recall / (precision + recall)
    else:
        fscore = 0
    
    return precision, recall, fscore

def evaluate_pair(test_file, gt_file, sample_points=200000):
    """Evaluate a single pair of point clouds"""
    print(f"  Loading {Path(test_file).name}...")
    test_mesh = trimesh.load(test_file, process=False)
    gt_mesh = trimesh.load(gt_file, process=False)
    
    test_pc = np.array(test_mesh.vertices)
    gt_pc = np.array(gt_mesh.vertices)
    
    print(f"    Test: {len(test_pc)} points")
    print(f"    GT:   {len(gt_pc)} points")
    
    # Compute metrics
    metrics = compute_chamfer_distance(test_pc, gt_pc, sample_points)
    
    # Compute precision/recall/fscore at different thresholds
    # UVG-CWI-DQPC data is in millimeters at full scale (~26 units mean distance)
    # Use thresholds appropriate for this scale: 10mm, 20mm, 30mm, 50mm
    for threshold in [10.0, 20.0, 30.0, 50.0]:
        p, r, f = compute_precision_recall_fscore(test_pc, gt_pc, threshold)
        metrics[f'precision_{threshold}'] = p
        metrics[f'recall_{threshold}'] = r
        metrics[f'fscore_{threshold}'] = f
    
    return metrics

def evaluate_directory_pairs(test_dir, gt_dir, output_csv, max_frames=None, sample_points=200000, file_suffix=''):
    """
    Evaluate all matching pairs in two directories.
    
    Args:
        test_dir: Directory with test files (CG or Reconstructed)
        gt_dir: Directory with ground truth files (HE)
        output_csv: Output CSV file
        max_frames: Maximum number of frames to evaluate
        sample_points: Number of points to sample
        file_suffix: Suffix to match in test_dir (e.g., '_pointcloud' for reconstructed)
    """
    test_dir = Path(test_dir)
    gt_dir = Path(gt_dir)
    
    # Get all test files
    if file_suffix:
        # For reconstructed: match frame_XXXX_pointcloud.ply pattern
        test_files = sorted(test_dir.glob(f'*{file_suffix}.ply'))
    else:
        # For baseline: match all .ply files
        test_files = sorted(test_dir.glob('*.ply'))
    
    gt_files = sorted(gt_dir.glob('*.ply'))
    
    if len(test_files) == 0:
        print(f"Error: No PLY files found in {test_dir}")
        return
    
    if max_frames:
        test_files = test_files[:max_frames]
        gt_files = gt_files[:max_frames]
    
    print(f"\nEvaluating {len(test_files)} pairs")
    print(f"Test dir: {test_dir}")
    print(f"GT dir:   {gt_dir}")
    print(f"Output:   {output_csv}")
    print()
    
    results = []
    
    for i, (test_file, gt_file) in enumerate(zip(test_files, gt_files)):
        print(f"[{i+1}/{len(test_files)}] Processing frame {i}...")
        
        try:
            metrics = evaluate_pair(test_file, gt_file, sample_points)
            result = {
                'frame': i,
                'test_file': test_file.name,
                'gt_file': gt_file.name,
                **metrics
            }
            results.append(result)
            
            print(f"    Chamfer: {metrics['chamfer_distance']:.6f}")
            print(f"    F-score@10mm: {metrics['fscore_10.0']:.4f}")
            print()
            
        except Exception as e:
            print(f"    Error: {e}")
            continue
    
    # Save results
    if results:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        
        fieldnames = ['frame', 'test_file', 'gt_file', 'chamfer_distance', 'accuracy', 'completeness', 
                     'hausdorff_distance', 'precision_10.0', 'recall_10.0', 'fscore_10.0',
                     'precision_20.0', 'recall_20.0', 'fscore_20.0',
                     'precision_30.0', 'recall_30.0', 'fscore_30.0',
                     'precision_50.0', 'recall_50.0', 'fscore_50.0']
        
        with open(output_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        
        print(f"✓ Results saved to {output_csv}")
        
        # Print summary
        avg_chamfer = np.mean([r['chamfer_distance'] for r in results])
        avg_fscore_10 = np.mean([r['fscore_10.0'] for r in results])
        avg_fscore_20 = np.mean([r['fscore_20.0'] for r in results])
        
        print(f"\nSummary ({len(results)} frames):")
        print(f"  Average Chamfer Distance: {avg_chamfer:.6f}")
        print(f"  Average F-score@10mm:     {avg_fscore_10:.4f}")
        print(f"  Average F-score@20mm:     {avg_fscore_20:.4f}")

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Evaluate OrangeKettlebell point clouds')
    parser.add_argument('--mode', type=str, required=True, choices=['baseline', 'reconstructed'],
                       help='Evaluation mode: baseline (CG vs HE) or reconstructed (reconstructed vs HE)')
    parser.add_argument('--cg_dir', type=str,
                       default='dataset/UVG-CWI-DQPC/OrangeKettlebell/CG/15fps',
                       help='Low-quality CG directory')
    parser.add_argument('--he_dir', type=str,
                       default='dataset/UVG-CWI-DQPC/OrangeKettlebell/HE/15fps',
                       help='Ground truth HE directory')
    parser.add_argument('--reconstructed_dir', type=str,
                       default='dataset/UVG-CWI-DQPC/OrangeKettlebell/Reconstructed',
                       help='Reconstructed directory (for reconstructed mode)')
    parser.add_argument('--max_frames', type=int, default=None,
                       help='Maximum number of frames to evaluate (default: all)')
    parser.add_argument('--sample_points', type=int, default=200000,
                       help='Number of points to sample for evaluation')
    
    args = parser.parse_args()
    
    print(f"\n{'='*70}")
    print(f"OrangeKettlebell Evaluation - {args.mode.upper()} MODE")
    print(f"{'='*70}\n")
    
    if args.mode == 'baseline':
        output_csv = Path('results/orangekettlebell_baseline_metrics.csv')
        evaluate_directory_pairs(args.cg_dir, args.he_dir, output_csv, 
                                args.max_frames, args.sample_points)
    
    elif args.mode == 'reconstructed':
        reconstructed_dir = Path(args.reconstructed_dir)
        if not reconstructed_dir.exists() or not any(reconstructed_dir.glob('*_pointcloud.ply')):
            print(f"Error: No reconstructed point cloud files found in {reconstructed_dir}")
            print("Expected files: frame_XXXX_pointcloud.ply")
            print("Please run SAL reconstruction first!")
            print("  python scripts/train_sal_single_frame.py 0 --epochs 500")
            return 1
        
        output_csv = Path('results/orangekettlebell_reconstructed_metrics.csv')
        evaluate_directory_pairs(args.reconstructed_dir, args.he_dir, output_csv,
                                args.max_frames, args.sample_points, file_suffix='_pointcloud')
    
    print(f"\n{'='*70}")
    print("Evaluation Complete!")
    print(f"{'='*70}\n")

if __name__ == '__main__':
    main()
