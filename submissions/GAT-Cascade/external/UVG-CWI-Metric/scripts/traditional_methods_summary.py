"""
Summary of Traditional Reconstruction Methods Results
Compares baseline (original CG) vs SPSR reconstruction.
No deep learning methods included.
"""

import pandas as pd
from pathlib import Path


def print_results():
    print("="*80)
    print("TRADITIONAL RECONSTRUCTION METHODS - RESULTS SUMMARY")
    print("="*80)
    print()
    
    # Load baseline
    baseline_file = Path("results/orangekettlebell_traditional_methods_baseline.csv")
    if not baseline_file.exists():
        print("ERROR: Baseline file not found!")
        return
    
    baseline_df = pd.read_csv(baseline_file)
    
    # Load reconstructed
    reconstructed_file = Path("results/orangekettlebell_traditional_methods_reconstructed.csv")
    if not reconstructed_file.exists():
        print("⚠️  Reconstructed file not yet available.")
        print("   Run evaluation first to generate results.")
        print()
        reconstructed_df = None
    else:
        reconstructed_df = pd.read_csv(reconstructed_file)
    
    print(f"Dataset: OrangeKettlebell (10 frames)")
    print(f"Methods: Baseline (CG) vs SPSR (Screened Poisson)")
    print()
    print("-"*80)
    print("BASELINE (Original Low-Quality CG Point Clouds)")
    print("-"*80)
    print(f"  Chamfer Distance:  {baseline_df['chamfer_distance'].mean():.2f} mm  (lower is better)")
    print(f"  F-score @ 10mm:    {baseline_df['fscore_10.0'].mean():.4f}      (higher is better)")
    print(f"  F-score @ 20mm:    {baseline_df['fscore_20.0'].mean():.4f}")
    print(f"  F-score @ 30mm:    {baseline_df['fscore_30.0'].mean():.4f}")
    print(f"  F-score @ 50mm:    {baseline_df['fscore_50.0'].mean():.4f}")
    print()
    
    if reconstructed_df is not None:
        print("-"*80)
        print("SPSR (Screened Poisson Surface Reconstruction)")
        print("-"*80)
        print(f"  Chamfer Distance:  {reconstructed_df['chamfer_distance'].mean():.2f} mm")
        print(f"  F-score @ 10mm:    {reconstructed_df['fscore_10.0'].mean():.4f}")
        print(f"  F-score @ 20mm:    {reconstructed_df['fscore_20.0'].mean():.4f}")
        print(f"  F-score @ 30mm:    {reconstructed_df['fscore_30.0'].mean():.4f}")
        print(f"  F-score @ 50mm:    {reconstructed_df['fscore_50.0'].mean():.4f}")
        print()
        
        # Compute improvements
        chamfer_improv = ((baseline_df['chamfer_distance'].mean() - 
                          reconstructed_df['chamfer_distance'].mean()) / 
                         baseline_df['chamfer_distance'].mean() * 100)
        
        fscore_10_improv = ((reconstructed_df['fscore_10.0'].mean() - 
                            baseline_df['fscore_10.0'].mean()) / 
                           baseline_df['fscore_10.0'].mean() * 100)
        
        fscore_20_improv = ((reconstructed_df['fscore_20.0'].mean() - 
                            baseline_df['fscore_20.0'].mean()) / 
                           baseline_df['fscore_20.0'].mean() * 100)
        
        print("-"*80)
        print("IMPROVEMENT (SPSR vs Baseline)")
        print("-"*80)
        print(f"  Chamfer Distance:  {chamfer_improv:+.1f}%")
        print(f"  F-score @ 10mm:    {fscore_10_improv:+.1f}%")
        print(f"  F-score @ 20mm:    {fscore_20_improv:+.1f}%")
        print()
        
        if chamfer_improv > 0:
            print("✅ SPSR reconstruction improved the point cloud quality!")
        else:
            print("⚠️  SPSR did not improve quality significantly.")
            print("   This can happen if the original CG is already reasonably good")
            print("   or if the reconstruction parameters need tuning.")
    
    print("="*80)
    print()
    print("Files:")
    print(f"  Baseline:      {baseline_file}")
    if reconstructed_df is not None:
        print(f"  Reconstructed: {reconstructed_file}")
    print()


if __name__ == "__main__":
    print_results()
