import sys
import subprocess
from pathlib import Path

# List of sequence names to process
SEQUENCES = [
    "BlueSpeech",
    "BouncingBlue",
    "FitFluencer",
    "Mannequin",
    "PinkNoir",
    "TrumanShow",
    "VirtualLife",
    "BlueVolley",
    "GoodVision",
    "OrangeKettlebell",
    "TicTacToe",
    "VictoryHeart"
]

def main():
    # Setup paths
    current_script = Path(__file__)
    script_dir = current_script.parent
    project_root = script_dir.parent
    
    # Path to the apply_transform.py script
    transform_script = script_dir / "apply_transform.py"
    
    # Base directory for datasets
    # Based on your workspace structure: dataset/UVG-CWI-DQPC/{Sequence}
    dataset_dir = project_root / "dataset" / "UVG-CWI-DQPC"
    
    # Directory where matrix .txt files are stored
    # Assuming they are in 'matrices' folder in project root, or change this path
    matrix_dir = project_root / "matrices" 
    
    print(f"Batch Processing Sequences...")
    print(f"Dataset root: {dataset_dir}")
    print(f"Matrix dir:   {matrix_dir}")
    print("-" * 50)

    for seq in SEQUENCES:
        print(f"Processing sequence: {seq}")

        # Define specific paths for this sequence
        # We try to detect the structure. 
        # Structure seen in workspace: {seq}/CG/15fps
        seq_path = dataset_dir / seq
        input_dir = seq_path / "CG" / "15fps"
        
        # If the simple structure doesn't exist, try the one from the bash script
        # {seq}/consumer-grade_capture_system/CG/15fps
        if not input_dir.exists():
            input_dir_alt = seq_path / "consumer-grade_capture_system" / "CG" / "15fps"
            if input_dir_alt.exists():
                input_dir = input_dir_alt

        # Output directory (aligned point clouds)
        # We'll save it next to the CG folder as "CG_aligned"
        output_dir = input_dir.parents[1] / "CG_aligned" / "15fps"
        
        # Matrix file
        matrix_file = matrix_dir / f"{seq}.txt"

        # Validation
        if not input_dir.exists():
            print(f"  [Skipping] Input directory not found: {input_dir}")
            continue
            
        if not matrix_file.exists():
            print(f"  [Skipping] Matrix file not found: {matrix_file}")
            # Optional: Warning only? No, we need the matrix.
            continue
            
        print(f"  Input:  {input_dir}")
        print(f"  Output: {output_dir}")
        print(f"  Matrix: {matrix_file}")
        
        # Construct the command
        cmd = [
            sys.executable, 
            str(transform_script),
            str(input_dir),
            str(output_dir),
            "--matrix_file", str(matrix_file)
        ]
        
        # Run the command
        try:
            # Running synchronous subprocess
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                print(f"  [Success] {seq} processed.")
            else:
                print(f"  [Error] {seq} failed.")
                print("  Error output:")
                print(result.stderr)
                
        except Exception as e:
            print(f"  [Exception] Failed to run script: {e}")

        print("-" * 30)

if __name__ == "__main__":
    main()
