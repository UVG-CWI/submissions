"""
Save baseline and reconstructed results to separate CSV files.
This script copies and renames the evaluation results to match your naming requirements.
"""

import pandas as pd
import shutil
from pathlib import Path


def save_baseline_csv():
    """
    Copy baseline metrics to results/orangekettlebell_traditional_methods_baseline.csv
    """
    baseline_excel = Path("results/orangekettlebell_baseline_metrics.xlsx")
    baseline_csv = Path("results/orangekettlebell_traditional_methods_baseline.csv")
    
    if baseline_excel.exists():
        # Read Excel and save as CSV
        df = pd.read_excel(baseline_excel)
        df.to_csv(baseline_csv, index=False)
        print(f"✓ Saved baseline results to: {baseline_csv}")
        print(f"  Columns: {list(df.columns)}")
        print(f"  Rows: {len(df)}")
        return df
    else:
        print(f"ERROR: Baseline Excel file not found: {baseline_excel}")
        return None


def save_reconstructed_csv():
    """
    Copy reconstructed metrics to results/orangekettlebell_traditional_methods_reconstructed.csv
    """
    # Check for SPSR results
    reconstructed_csv_source = Path("results/orangekettlebell_reconstructed_metrics.csv")
    reconstructed_csv_dest = Path("results/orangekettlebell_traditional_methods_reconstructed.csv")
    
    if reconstructed_csv_source.exists():
        # Read and save
        df = pd.read_csv(reconstructed_csv_source)
        df.to_csv(reconstructed_csv_dest, index=False)
        print(f"\n✓ Saved reconstructed results to: {reconstructed_csv_dest}")
        print(f"  Columns: {list(df.columns)}")
        print(f"  Rows: {len(df)}")
        return df
    else:
        print(f"\nERROR: Reconstructed CSV file not found: {reconstructed_csv_source}")
        return None


def print_summary(baseline_df, reconstructed_df):
    """
    Print comparison summary.
    """
    if baseline_df is not None and reconstructed_df is not None:
        print("\n" + "="*80)
        print("COMPARISON SUMMARY")
        print("="*80)
        
        # Baseline stats
        print("\nBASELINE (Original CG low-quality):")
        print(f"  Chamfer Distance: {baseline_df['chamfer_distance'].mean():.2f} mm")
        print(f"  F-score@10mm:     {baseline_df['fscore_10.0'].mean():.4f}")
        print(f"  F-score@20mm:     {baseline_df['fscore_20.0'].mean():.4f}")
        print(f"  F-score@30mm:     {baseline_df['fscore_30.0'].mean():.4f}")
        
        # Reconstructed stats
        print("\nRECONSTRUCTED (After SPSR):")
        print(f"  Chamfer Distance: {reconstructed_df['chamfer_distance'].mean():.2f} mm")
        print(f"  F-score@10mm:     {reconstructed_df['fscore_10.0'].mean():.4f}")
        print(f"  F-score@20mm:     {reconstructed_df['fscore_20.0'].mean():.4f}")
        print(f"  F-score@30mm:     {reconstructed_df['fscore_30.0'].mean():.4f}")
        
        # Improvement
        chamfer_improv = ((baseline_df['chamfer_distance'].mean() - reconstructed_df['chamfer_distance'].mean()) / 
                         baseline_df['chamfer_distance'].mean() * 100)
        fscore_improv = ((reconstructed_df['fscore_10.0'].mean() - baseline_df['fscore_10.0'].mean()) / 
                        baseline_df['fscore_10.0'].mean() * 100)
        
        print("\nIMPROVEMENT:")
        print(f"  Chamfer:      {chamfer_improv:+.1f}% (lower is better)")
        print(f"  F-score@10mm: {fscore_improv:+.1f}% (higher is better)")
        print("="*80)


def main():
    print("="*80)
    print("SAVING SEPARATE CSV FILES")
    print("="*80)
    
    # Save baseline
    baseline_df = save_baseline_csv()
    
    # Save reconstructed
    reconstructed_df = save_reconstructed_csv()
    
    # Print summary
    print_summary(baseline_df, reconstructed_df)
    
    print("\n✓ Done!")


if __name__ == "__main__":
    main()
