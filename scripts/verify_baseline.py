import cv2
import numpy as np
from ultralytics import YOLO

def verify_baseline():
    print("Loading YOLOv8n model...")
    try:
        model = YOLO('yolov8n.pt')  # This will download the weights if not present
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    # Create a dummy image (black square)
    img = np.zeros((640, 640, 3), dtype=np.uint8)
    cv2.putText(img, 'Test', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    print("Running inference on dummy image...")
    try:
        results = model(img)
        print("Inference successful.")
        print(f"Detected {len(results[0].boxes)} objects (expected 0 or few on dummy).")
    except Exception as e:
        print(f"Inference failed: {e}")
        return

    print("Baseline verification PASSED.")

if __name__ == "__main__":
    verify_baseline()
