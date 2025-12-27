"""
Convert TRANCOS dataset annotations to JSON format for PSDDN training.

TRANCOS usually provides annotations in text files (one per image) containing
coordinates of vehicles, or sometimes as a 'dots' image.
This script assumes the common text format:
    x1 y1
    x2 y2
    ...
Or CSV format.

Usage:
    python scripts/convert_trancos.py --data-dir data/TRANCOS --output data/trancos/annotations.json
"""

import os
import json
import argparse
from pathlib import Path
import glob

def convert_trancos_to_json(data_dir, output_file):
    annotations = {}
    data_path = Path(data_dir)
    
    # TRANCOS typically has an 'images' folder and ... annotations?
    # Let's search recursively for text files that might be annotations
    # excluding standard files like README, etc.
    
    # Common convention: image_name.txt matches image_name.jpg
    image_files = sorted(list(data_path.rglob("*.jpg")))
    
    print(f"Found {len(image_files)} images in {data_dir}")
    
    count = 0
    for img_path in image_files:
        # Look for corresponding text file
        txt_path = img_path.with_suffix('.txt')
        
        if not txt_path.exists():
            # Try looking in an 'annotations' or 'labels' sibling folder
            # e.g. images/X.jpg -> labels/X.txt
            parts = list(img_path.parts)
            if 'images' in parts:
                idx = parts.index('images')
                parts[idx] = 'dots' # common in TRANCOS to have 'dots' folder? or just text
                # Actually TRANCOS often provides `vectmap.txt` or similar.
                # Let's assume standard .txt format for now.
                pass
                
        if txt_path.exists():
            points = []
            with open(txt_path, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    try:
                        # Parse x, y
                        parts = line.strip().split()
                        if len(parts) >= 2:
                            x, y = float(parts[0]), float(parts[1])
                            points.append([x, y])
                    except ValueError:
                        pass
            
            if points:
                # Store with relative path or filename
                # PSDDN pipeline expects filename key usually
                annotations[img_path.name] = points
                count += 1
                
    if count == 0:
        print("WARNING: No text annotation files found matching image names!")
        print("Required format: image.jpg and image.txt (with x y lines)")
        print("Searching for 'dots.png' files instead...")
        
        # Alternative: Dots images
        # Requires opencv
        try:
            import cv2
            import numpy as np
            
            dots_files = sorted(list(data_path.rglob("*dots.png")))
            print(f"Found {len(dots_files)} dots.png files")
            
            for dot_path in dots_files:
                # corresponding image
                img_name = dot_path.name.replace('dots.png', '.jpg') # Guessing naming convention
                
                img = cv2.imread(str(dot_path), cv2.IMREAD_GRAYSCALE)
                if img is None: continue
                
                # Find centroids of dots
                # binary threshold
                _, thresh = cv2.threshold(img, 10, 255, cv2.THRESH_BINARY)
                contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                points = []
                for c in contours:
                    M = cv2.moments(c)
                    if M["m00"] != 0:
                        cX = M["m10"] / M["m00"]
                        cY = M["m01"] / M["m00"]
                        points.append([cX, cY])
                
                if points:
                    annotations[img_name] = points
                    count += 1
                    
        except ImportError:
            print("OpenCV not installed, cannot process dots.png files.")
            
    print(f"Converted {count} annotated images.")
    
    # Save to JSON
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(annotations, f, indent=2)
    print(f"Saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', required=True, help='Root directory of TRANCOS data')
    parser.add_argument('--output', required=True, help='Output JSON file')
    args = parser.parse_args()
    
    convert_trancos_to_json(args.data_dir, args.output)
