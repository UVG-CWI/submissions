# Compare two files (overlapped)
#python scripts/compare_ply_o3d.py path/to/file1.ply path/to/file2.ply
# Compare side-by-side (using offset, e.g., 50 units apart)
#python scripts/compare_ply_o3d.py path/to/file1.ply path/to/file2.ply --offset 50

import open3d as o3d
import argparse
import numpy as np
import sys

def load_and_paint(path, color):
    """
    Load a point cloud and paint it a uniform color.
    path: str path to ply
    color: list [r,g,b] float 0-1
    """
    print(f"Loading {path}...")
    try:
        pcd = o3d.io.read_point_cloud(path)
        if pcd.is_empty():
            print(f"Warning: {path} is empty or could not be loaded.")
            return None
        
        # Paint uniform color
        pcd.paint_uniform_color(color)
        return pcd
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Compare two PLY files using Open3D")
    parser.add_argument("file1", help="Path to the first PLY file (Red)")
    parser.add_argument("file2", help="Path to the second PLY file (Blue)")
    parser.add_argument("--offset", type=float, default=0.0, help="Offset the second cloud in X axis")
    
    args = parser.parse_args()
    
    # Red for file1, Blue for file2
    # Open3D uses float colors [0, 1]
    pcd1 = load_and_paint(args.file1, [1, 0, 0])
    pcd2 = load_and_paint(args.file2, [0, 0, 1])
    
    if not pcd1 or not pcd2:
        return

    # Apply offset if needed
    if args.offset != 0.0:
        print(f"Applying X offset: {args.offset}")
        pcd2.translate((args.offset, 0, 0))
        
    print("\nVisualizing...")
    print("Red:  " + args.file1)
    print("Blue: " + args.file2)
    print("\nControls:")
    print("  - Mouse Drag: Rotate")
    print("  - Ctrl + Mouse Drag: Pan")
    print("  - Mouse Wheel: Zoom")
    print("  - +/-: Increase/Decrease Point Size")
    print("  - 'q': Quit")
    
    # Visualize
    try:
        o3d.visualization.draw_geometries([pcd1, pcd2], 
                                          window_name="PLY Comparison",
                                          width=1024,
                                          height=768)
    except Exception as e:
        print(f"Error opening viewer: {e}")

if __name__ == "__main__":
    main()
