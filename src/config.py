from pathlib import Path
import torch

# Base paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"

# Dataset configurations
DATASET_CONFIG = {
    "train_path": str(DATA_DIR / "train"),
    "val_path": str(DATA_DIR / "val"),
    "test_path": str(DATA_DIR / "test"),
    "num_classes": 80,  # COCO dataset default
    "class_names": None  # Will be loaded from dataset
}

# Training configurations
TRAIN_CONFIG = {
    "batch_size": 16,
    "epochs": 100,
    "image_size": 640,
    "device": torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    "workers": 4,
    "optimizer": "SGD",
    "learning_rate": 0.01,
    "momentum": 0.937,
    "weight_decay": 0.0005,
}

# YOLOv8 specific configurations
YOLOV8_CONFIG = {
    "model_type": "yolov8s",  # yolov8n, yolov8s, yolov8m, yolov8l, yolov8x
    "pretrained": True,
    "conf_thres": 0.25,
    "iou_thres": 0.45,
}

# YOLOv9 specific configurations
YOLOV9_CONFIG = {
    "model_type": "yolov9s",  # yolov9-c, yolov9-e
    "pretrained": True,
    "conf_thres": 0.25,
    "iou_thres": 0.45,
}

# Evaluation configurations
EVAL_CONFIG = {
    "metrics": ["mAP50", "mAP50-95", "precision", "recall"],
    "save_results": True,
    "visualization": True,
} 