import os
from ultralytics import YOLO

def train_model():
    """Trains the YOLOv8 model for PPE Detection."""
    dataset_yaml = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data/dataset/dataset.yaml'))
    
    # Load a pre-trained nano model
    model = YOLO('yolov8n.pt')

    print(f"Starting training using dataset config: {dataset_yaml}")
    
    # Train the model
    results = model.train(
        data=dataset_yaml,
        epochs=50,
        imgsz=640,
        batch=16,
        project='../../models',
        name='yolov8_ppe',
        exist_ok=True
    )
    
    print("Training completed. Model saved to models/yolov8_ppe/weights/best.pt")

if __name__ == "__main__":
    train_model()
