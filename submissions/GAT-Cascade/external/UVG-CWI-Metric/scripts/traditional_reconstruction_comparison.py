"""
Traditional Surface Reconstruction Methods Comparison
Based on the survey paper methods (non-deep-learning):

1. SPSR (Screened Poisson Surface Reconstruction) - Kazhdan & Hoppe 2013
2. BPA (Ball Pivoting Algorithm) - Bernardini et al. 1999

This script compares traditional reconstruction methods against SAL (deep learning).
"""

import argparse
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import subprocess
import sys


def run_poisson_reconstruction(frames=10, depth=8, knn=30):
    """Run SPSR (Screened Poisson) reconstruction."""
    print("\n" + "="*80)
    print("Running SPSR (Screened Poisson Surface Reconstruction)")
    print("="*80)
    
    cmd = [
        sys.executable,
        "scripts/poisson_reconstruction.py",
        "--method", "poisson",
        "--frames", str(frames),
        "--depth", str(depth),
        "--knn", str(knn)
    ]
    
    result = subprocess.run(cmd, capture_output=False, text=True)
    return result.returncode == 0


def run_bpa_reconstruction(frames=10, knn=30):
    """Run BPA (Ball Pivoting Algorithm) reconstruction."""
    print("\n" + "="*80)
    print("Running BPA (Ball Pivoting Algorithm)")
    print("="*80)
    
    cmd = [
        sys.executable,
        "scripts/poisson_reconstruction.py",
        "--method", "bpa",
        "--frames", str(frames),
        "--knn", str(knn)
    ]
    
    result = subprocess.run(cmd, capture_output=False, text=True)
    return result.returncode == 0


def evaluate_reconstructed(frames=10, method_name="reconstructed"):
    """Evaluate reconstructed point clouds."""
    print("\n" + "="*80)
    print(f"Evaluating {method_name} results")
    print("="*80)
    
    cmd = [
        sys.executable,
        "scripts/evaluate_orangekettlebell.py",
        "--mode", "reconstructed",
        "--max_frames", str(frames)
    ]
    
    result = subprocess.run(cmd, capture_output=False, text=True)
    return result.returncode == 0


def save_results_to_excel(method_name, output_dir="results"):
    """Save evaluation results to Excel with method-specific naming."""
    import pandas as pd
    from pathlib import Path
    
    # Read CSV results
    csv_path = Path("results") / "orangekettlebell_reconstructed_metrics.csv"
    if not csv_path.exists():
        print(f"Warning: Results CSV not found at {csv_path}")
        return
    
    df = pd.read_csv(csv_path)
    
    # Save to Excel with method name
    output_path = Path(output_dir) / f"orangekettlebell_{method_name}_metrics.xlsx"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    df.to_excel(output_path, index=False)
    print(f"Saved {method_name} results to: {output_path}")


def compare_all_methods(frames=10):
    """
    Compare all reconstruction methods:
    - Baseline (no reconstruction)
    - SPSR (Screened Poisson)
    - BPA (Ball Pivoting Algorithm)
    - SAL (deep learning - if already run)
    """
    print("\n" + "="*80)
    print("TRADITIONAL RECONSTRUCTION METHODS COMPARISON")
    print("="*80)
    print(f"\nProcessing {frames} frames")
    print("\nMethods to compare:")
    print("1. Baseline (original low-quality CG)")
    print("2. SPSR (Screened Poisson Surface Reconstruction)")
    print("3. BPA (Ball Pivoting Algorithm)")
    print("4. SAL (Sign Agnostic Learning - if available)")
    
    results_summary = []
    
    # 1. Baseline evaluation (if not already done)
    baseline_path = Path("results/orangekettlebell_baseline_metrics.xlsx")
    if not baseline_path.exists():
        print("\n--- Running baseline evaluation ---")
        cmd = [sys.executable, "scripts/evaluate_orangekettlebell.py", 
               "--mode", "baseline", "--max_frames", str(frames)]
        subprocess.run(cmd)
        
        # Save baseline to Excel
        csv_path = Path("results/orangekettlebell_baseline_metrics.csv")
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            df.to_excel(baseline_path, index=False)
    
    # Load baseline results
    if baseline_path.exists():
        baseline_df = pd.read_excel(baseline_path)
        baseline_summary = {
            'Method': 'Baseline (CG)',
            'Chamfer_Distance': baseline_df['chamfer_distance'].mean(),
            'F-score@10mm': baseline_df['fscore_10.0'].mean(),
            'F-score@20mm': baseline_df['fscore_20.0'].mean(),
            'F-score@30mm': baseline_df['fscore_30.0'].mean(),
        }
        results_summary.append(baseline_summary)
        print(f"\nBaseline: Chamfer={baseline_summary['Chamfer_Distance']:.2f}mm, "
              f"F@10mm={baseline_summary['F-score@10mm']:.4f}")
    
    # 2. SPSR reconstruction
    print("\n--- Running SPSR reconstruction ---")
    if run_poisson_reconstruction(frames=frames, depth=9, knn=30):
        if evaluate_reconstructed(frames=frames, method_name="SPSR"):
            save_results_to_excel("spsr")
            
            spsr_path = Path("results/orangekettlebell_spsr_metrics.xlsx")
            if spsr_path.exists():
                spsr_df = pd.read_excel(spsr_path)
                spsr_summary = {
                    'Method': 'SPSR (Poisson)',
                    'Chamfer_Distance': spsr_df['chamfer_distance'].mean(),
                    'F-score@10mm': spsr_df['fscore_10.0'].mean(),
                    'F-score@20mm': spsr_df['fscore_20.0'].mean(),
                    'F-score@30mm': spsr_df['fscore_30.0'].mean(),
                }
                results_summary.append(spsr_summary)
                print(f"\nSPSR: Chamfer={spsr_summary['Chamfer_Distance']:.2f}mm, "
                      f"F@10mm={spsr_summary['F-score@10mm']:.4f}")
    
    # 3. BPA reconstruction
    print("\n--- Running BPA reconstruction ---")
    if run_bpa_reconstruction(frames=frames, knn=30):
        if evaluate_reconstructed(frames=frames, method_name="BPA"):
            save_results_to_excel("bpa")
            
            bpa_path = Path("results/orangekettlebell_bpa_metrics.xlsx")
            if bpa_path.exists():
                bpa_df = pd.read_excel(bpa_path)
                bpa_summary = {
                    'Method': 'BPA (Ball Pivot)',
                    'Chamfer_Distance': bpa_df['chamfer_distance'].mean(),
                    'F-score@10mm': bpa_df['fscore_10.0'].mean(),
                    'F-score@20mm': bpa_df['fscore_20.0'].mean(),
                    'F-score@30mm': bpa_df['fscore_30.0'].mean(),
                }
                results_summary.append(bpa_summary)
                print(f"\nBPA: Chamfer={bpa_summary['Chamfer_Distance']:.2f}mm, "
                      f"F@10mm={bpa_summary['F-score@10mm']:.4f}")
    
    # 4. Check for SAL results (if already available)
    sal_path = Path("results/orangekettlebell_sal_metrics.xlsx")
    if sal_path.exists():
        sal_df = pd.read_excel(sal_path)
        sal_summary = {
            'Method': 'SAL (Deep Learning)',
            'Chamfer_Distance': sal_df['chamfer_distance'].mean(),
            'F-score@10mm': sal_df['fscore_10.0'].mean(),
            'F-score@20mm': sal_df['fscore_20.0'].mean(),
            'F-score@30mm': sal_df['fscore_30.0'].mean(),
        }
        results_summary.append(sal_summary)
        print(f"\nSAL: Chamfer={sal_summary['Chamfer_Distance']:.2f}mm, "
              f"F@10mm={sal_summary['F-score@10mm']:.4f}")
    
    # Create comparison table
    if results_summary:
        comparison_df = pd.DataFrame(results_summary)
        
        # Save comparison
        comparison_path = Path("results/orangekettlebell_methods_comparison.xlsx")
        comparison_df.to_excel(comparison_path, index=False)
        
        print("\n" + "="*80)
        print("FINAL COMPARISON")
        print("="*80)
        print(comparison_df.to_string(index=False))
        print(f"\nComparison saved to: {comparison_path}")
        
        # Highlight improvements
        if len(results_summary) > 1:
            baseline_chamfer = results_summary[0]['Chamfer_Distance']
            print("\n--- Improvements over Baseline ---")
            for i, result in enumerate(results_summary[1:], 1):
                chamfer_improvement = ((baseline_chamfer - result['Chamfer_Distance']) / baseline_chamfer) * 100
                print(f"{result['Method']}: {chamfer_improvement:+.1f}% Chamfer distance")


def main():
    parser = argparse.ArgumentParser(
        description="Compare traditional reconstruction methods (SPSR, BPA) vs baseline"
    )
    parser.add_argument('--frames', type=int, default=10,
                       help='Number of frames to process (default: 10)')
    parser.add_argument('--method', type=str, choices=['all', 'spsr', 'bpa'],
                       default='all', help='Which method to run (default: all)')
    parser.add_argument('--poisson_depth', type=int, default=9,
                       help='Poisson octree depth for SPSR (default: 9)')
    parser.add_argument('--knn', type=int, default=30,
                       help='KNN for normal estimation (default: 30)')
    
    args = parser.parse_args()
    
    if args.method == 'all':
        compare_all_methods(frames=args.frames)
    elif args.method == 'spsr':
        if run_poisson_reconstruction(frames=args.frames, depth=args.poisson_depth, knn=args.knn):
            evaluate_reconstructed(frames=args.frames, method_name="SPSR")
            save_results_to_excel("spsr")
    elif args.method == 'bpa':
        if run_bpa_reconstruction(frames=args.frames, knn=args.knn):
            evaluate_reconstructed(frames=args.frames, method_name="BPA")
            save_results_to_excel("bpa")


if __name__ == "__main__":
    main()
