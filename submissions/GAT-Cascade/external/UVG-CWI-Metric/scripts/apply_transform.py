import open3d as o3d
import argparse
import numpy as np
import sys
from pathlib import Path

def parse_matrix_arg(matrix_str):
    """
    Parse a string of 16 numbers into a 4x4 numpy matrix.
    """
    try:
        # standardizing delimiters: replace commas with spaces
        cleaned = matrix_str.replace(',', ' ').strip().split()
        values = [float(x) for x in cleaned]
        if len(values) != 16:
            raise ValueError(f"Expected 16 values, got {len(values)}")
        return np.array(values).reshape(4, 4)
    except Exception as e:
        print(f"Error parsing matrix string: {e}")
        return None

def load_matrix_file(path):
    """Load matrix from a text file."""
    try:
        mat = np.loadtxt(path)
        if mat.size != 16:
             raise ValueError(f"Expected 16 values in file, got {mat.size}")
        return mat.reshape(4, 4)
    except Exception as e:
        print(f"Error loading matrix file: {e}")
        return None

def transform_and_save(input_file, output_file, transform):
    """Applies transform to a single file."""
    print(f"Processing: {input_file.name} -> {output_file.name}")
    try:
        # Load
        pcd = o3d.io.read_point_cloud(str(input_file))
        if pcd.is_empty():
            print(f"  Warning: {input_file} is empty or invalid format. Skipping.")
            return False
            
        # Transform
        pcd.transform(transform)
        
        # Save
        # Binary PLY is faster and smaller
        success = o3d.io.write_point_cloud(str(output_file), pcd, write_ascii=False)
        if not success:
            print(f"  Error: Failed to write to {output_file}")
            return False
            
        return True
    except Exception as e:
        print(f"  Error processing file: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Apply 4x4 transformation matrix to a PLY file OR a directory of PLY files.")
    parser.add_argument("input_path", help="Input PLY file path OR directory containing PLY files")
    parser.add_argument("output_path", help="Output PLY file path OR output directory")
    
    # Transformation source
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--matrix", help="4x4 transformation matrix as a string (16 floats).")
    group.add_argument("--matrix_file", help="Path to a text file containing the 4x4 matrix values.")

    args = parser.parse_args()
    
    # 1. Parse Matrix
    transform = None
    if args.matrix:
        transform = parse_matrix_arg(args.matrix)
    elif args.matrix_file:
        transform = load_matrix_file(args.matrix_file)
        
    if transform is None:
        sys.exit(1)
        
    print("Transformation Matrix:")
    print(transform)
    print("-" * 30)

    # 2. Determine Mode (Single File vs Directory)
    in_path = Path(args.input_path)
    out_path = Path(args.output_path)
    
    if not in_path.exists():
        print(f"Error: Input path '{in_path}' does not exist.")
        sys.exit(1)

    if in_path.is_dir():
        # --- Directory Mode ---
        print(f"Batch Mode: Directory '{in_path}'")
        
        # Ensure output directory exists
        if not out_path.exists():
            print(f"Creating output directory: {out_path}")
            out_path.mkdir(parents=True, exist_ok=True)
        elif not out_path.is_dir():
             print(f"Error: Input is a directory, but output '{out_path}' is a file.")
             sys.exit(1)
             
        # Find all .ply files
        ply_files = list(in_path.glob("*.ply"))
        if not ply_files:
            print("No .ply files found in input directory.")
            return

        print(f"Found {len(ply_files)} PLY files.")
        
        count = 0
        for f in ply_files:
            # Output filename is same as input filename
            dest = out_path / f.name
            if transform_and_save(f, dest, transform):
                count += 1
                
        print("-" * 30)
        print(f"Batch processing complete. Successfully processed {count}/{len(ply_files)} files.")

    else:
        # --- Single File Mode ---
        # Ensure output directory exists (if output path includes one)
        if out_path.parent and not out_path.parent.exists():
            out_path.parent.mkdir(parents=True, exist_ok=True)
            
        transform_and_save(in_path, out_path, transform)

if __name__ == "__main__":
    main()
