from ultralytics import YOLO
import torch
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np

from config import YOLOV8_CONFIG, TRAIN_CONFIG
from utils_codes import setup_logging

logger = setup_logging(__name__)

class YOLOv8Detector:
    def __init__(self, model_type: str = YOLOV8_CONFIG["model_type"],
                 pretrained: bool = YOLOV8_CONFIG["pretrained"]):
        """Initialize YOLOv8 detector."""
        self.model_type = model_type
        self.pretrained = pretrained
        self.model = self._load_model()
        self.device = torch.device(TRAIN_CONFIG["device"])
        self.model.to(self.device)
        
        logger.info(f"Initialized YOLOv8 detector with {model_type} model")
    
    def _load_model(self) -> YOLO:
        """Load YOLOv8 model."""
        if self.pretrained:
            model = YOLO(f"{self.model_type}.pt")
        else:
            model = YOLO(self.model_type)
        return model
    
    def train(self, 
              data_yaml: str,
              epochs: int = TRAIN_CONFIG["epochs"],
              batch_size: int = TRAIN_CONFIG["batch_size"],
              image_size: int = TRAIN_CONFIG["image_size"],
              workers: int = TRAIN_CONFIG["workers"]) -> Dict:
        """Train the model."""
        logger.info("Starting training...")
        
        results = self.model.train(
            data=data_yaml,
            epochs=epochs,
            batch=batch_size,
            imgsz=image_size,
            workers=workers,
            device=TRAIN_CONFIG["device"],
            optimizer=TRAIN_CONFIG["optimizer"],
            lr0=TRAIN_CONFIG["learning_rate"],
            momentum=TRAIN_CONFIG["momentum"],
            weight_decay=TRAIN_CONFIG["weight_decay"]
        )
        
        logger.info("Training completed")
        return results
    
    def evaluate(self, 
                data_yaml: str,
                batch_size: int = TRAIN_CONFIG["batch_size"],
                image_size: int = TRAIN_CONFIG["image_size"]) -> Dict:
        """Evaluate the model."""
        logger.info("Starting evaluation...")
        
        results = self.model.val(
            data=data_yaml,
            batch=batch_size,
            imgsz=image_size,
            device=TRAIN_CONFIG["device"]
        )
        
        logger.info("Evaluation completed")
        return results
    
    def predict(self, 
                image: np.ndarray,
                conf_thres: float = YOLOV8_CONFIG["conf_thres"],
                iou_thres: float = YOLOV8_CONFIG["iou_thres"]) -> Tuple[List[List[float]], List[int], List[float]]:
        """
        Perform object detection on an image.
        Returns: (boxes, labels, scores)
        """
        results = self.model.predict(
            source=image,
            conf=conf_thres,
            iou=iou_thres,
            device=TRAIN_CONFIG["device"]
        )[0]
        
        boxes = results.boxes.xyxy.cpu().numpy()
        labels = results.boxes.cls.cpu().numpy().astype(int)
        scores = results.boxes.conf.cpu().numpy()
        
        return boxes, labels, scores
    
    def export(self, format: str = "onnx", save_dir: Optional[str] = None):
        """Export the model to different formats."""
        if save_dir:
            Path(save_dir).mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Exporting model to {format} format...")
        self.model.export(format=format, save_dir=save_dir)
        logger.info("Export completed") 