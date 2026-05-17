import os
from ultralytics import YOLO

class YoloDetector:
    def __init__(self, model_path=None, conf_thresh=0.5):
        custom_model_path = "models/yolov8_ppe/weights/best.pt"
        pretrained_model_path = "models/yolov8_ppe/weights/pretrained.pt"
        
        # Try custom model first, then fallback to pretrained
        if model_path is None:
            if os.path.exists(custom_model_path):
                model_path = custom_model_path
            elif os.path.exists(pretrained_model_path):
                model_path = pretrained_model_path
            else:
                raise FileNotFoundError("No YOLOv8 weights found. Please train the model or download the pretrained weights.")
        
        try:
            self.model = YOLO(model_path)
            print(f"Successfully loaded model: {model_path}")
        except Exception as e:
            print(f"Failed to load custom model: {e}")
            raise e
            
        self.conf_thresh = conf_thresh

    def detect(self, frame):
        results = self.model.predict(frame, conf=self.conf_thresh, verbose=False)
        return results[0]

    def get_bboxes(self, result):
        boxes = result.boxes
        detections = []
        if boxes is None:
            return detections
            
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = box.conf[0].item()
            cls_id = int(box.cls[0].item())
            class_name = self.model.names[cls_id]
            detections.append({
                "bbox": [int(x1), int(y1), int(x2), int(y2)],
                "conf": conf,
                "class_id": cls_id,
                "class_name": class_name
            })
        return detections
