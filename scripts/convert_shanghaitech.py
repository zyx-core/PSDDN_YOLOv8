"""Convert ShanghaiTech MAT annotations to JSON format."""
import scipy.io as sio
import json
from pathlib import Path
import argparse


def convert_mat_to_json(mat_dir, output_file):
    """Convert all MAT files in directory to single JSON file."""
    annotations = {}
    mat_path = Path(mat_dir)
    
    # Try recursive glob if direct doesn't yield results
    mat_files = list(mat_path.glob("*.mat"))
    if not mat_files:
        mat_files = list(mat_path.rglob("*.mat"))
        
    print(f"Found {len(mat_files)} MAT files in {mat_dir}")
    
    # Check for image folder to verify filenames
    image_dir = mat_path.parent.parent / "images"
    has_processed_prefix = False
    if image_dir.exists():
        if list(image_dir.glob("processed_IMG_*.jpg")):
             has_processed_prefix = True
             print("Detected 'processed_' prefix in image files.")

    for mat_file in mat_files:
        try:
            # Load MAT file
            mat = sio.loadmat(str(mat_file))
            # Standard ShanghaiTech structure: image_info -> location
            if 'image_info' in mat:
                points = mat['image_info'][0, 0]['location'][0, 0]
            else:
                # Some versions might have different keys
                potential_keys = [k for k in mat.keys() if not k.startswith('__')]
                print(f"  Warning: 'image_info' not found in {mat_file.name}. Keys: {potential_keys}")
                continue
                
            # Image name (remove GT_ prefix and .mat extension)
            base_name = mat_file.stem.replace('GT_', '')
            
            # Adjust if images have prefix
            if has_processed_prefix:
                img_name = f"processed_{base_name}"
            else:
                img_name = base_name
            
            # Convert to list of [x, y] coordinates
            point_list = points.tolist()
            annotations[img_name] = point_list
            print(f"  {img_name}: {len(point_list)} points")
        except Exception as e:
            print(f"  Error processing {mat_file.name}: {e}")
    
    # Ensure output directory exists
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    
    # Save to JSON
    with open(output_file, 'w') as f:
        json.dump(annotations, f, indent=2)
    
    print(f"\nSaved annotations to {output_file}")
    print(f"Total images: {len(annotations)}")
    if annotations:
        print(f"Total points: {sum(len(pts) for pts in annotations.values())}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert ShanghaiTech MAT to JSON')
    parser.add_argument('--mat-dir', required=True, help='Directory with MAT files')
    parser.add_argument('--output', required=True, help='Output JSON file')
    args = parser.parse_args()
    
    convert_mat_to_json(args.mat_dir, args.output)
