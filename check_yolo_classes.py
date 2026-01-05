from ultralytics import YOLO

model_path = 'YOLO_detection/runs/detect/keypoint_detection/weights/best.pt'
try:
    model = YOLO(model_path)
    print("Model classes:")
    print(model.names)
except Exception as e:
    print(f"Error loading model: {e}")
