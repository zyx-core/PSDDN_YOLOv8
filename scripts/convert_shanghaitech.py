"""Convert ShanghaiTech MAT annotations to JSON format."""
import scipy.io as sio
import json
from pathlib import Path
import argparse


def convert_mat_to_json(mat_dir, output_file):
    """Convert all MAT files in directory to single JSON file."""
    annotations = {}
    
    mat_files = list(Path(mat_dir).glob("*.mat"))
    print(f"Found {len(mat_files)} MAT files")
    
    for mat_file in mat_files:
        # Load MAT file
        mat = sio.loadmat(str(mat_file))
        points = mat['image_info'][0, 0]['location'][0, 0]
        
        # Image name (remove GT_ prefix and .mat extension)
        img_name = mat_file.stem.replace('GT_', '')
        
        # Convert to list of [x, y] coordinates
        point_list = points.tolist()
        annotations[img_name] = point_list
        
        print(f"  {img_name}: {len(point_list)} points")
    
    # Save to JSON
    with open(output_file, 'w') as f:
        json.dump(annotations, f, indent=2)
    
    print(f"\nSaved annotations to {output_file}")
    print(f"Total images: {len(annotations)}")
    print(f"Total annotations: {sum(len(pts) for pts in annotations.values())}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert ShanghaiTech MAT to JSON')
    parser.add_argument('--mat-dir', required=True, help='Directory with MAT files')
    parser.add_argument('--output', required=True, help='Output JSON file')
    args = parser.parse_args()
    
    convert_mat_to_json(args.mat_dir, args.output)
