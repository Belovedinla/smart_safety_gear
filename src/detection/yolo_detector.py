import cv2
from ultralytics import YOLO

class YoloDetector:
    def __init__(self, model_path="models/yolov8_ppe/weights/best.pt", conf_thresh=0.5):
        try:
            self.model = YOLO(model_path)
        except Exception as e:
            print(f"Custom model not found: {e}. Loading standard yolov8n.pt")
            self.model = YOLO("yolov8n.pt")
            
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
