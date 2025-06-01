import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Tuple, Dict, Union
import logging
from datetime import datetime

def setup_logging(name: str) -> logging.Logger:
    """Set up logging configuration."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    
    # File handler
    log_file = f"logs/{datetime.now().strftime('%Y%m%d_%H%M%S')}_{name}.log"
    Path("logs").mkdir(exist_ok=True)
    fh = logging.FileHandler(log_file)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    
    return logger

def plot_training_results(results: Dict[str, List[float]], save_path: str = None):
    """Plot training metrics."""
    plt.figure(figsize=(15, 5))
    
    for i, (metric, values) in enumerate(results.items(), 1):
        plt.subplot(1, len(results), i)
        plt.plot(values)
        plt.title(f'{metric} over epochs')
        plt.xlabel('Epoch')
        plt.ylabel(metric)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.close()

def visualize_detection(image: np.ndarray, 
                       boxes: List[List[float]], 
                       labels: List[int],
                       scores: List[float],
                       class_names: List[str],
                       conf_threshold: float = 0.25) -> np.ndarray:
    """Visualize detection results on image."""
    image_copy = image.copy()
    
    for box, label, score in zip(boxes, labels, scores):
        if score < conf_threshold:
            continue
            
        x1, y1, x2, y2 = map(int, box)
        class_name = class_names[label]
        
        # Draw box
        cv2.rectangle(image_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Draw label
        label_text = f'{class_name}: {score:.2f}'
        cv2.putText(image_copy, label_text, (x1, y1 - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    return image_copy

def compute_iou(box1: List[float], box2: List[float]) -> float:
    """Compute IoU between two bounding boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    union = box1_area + box2_area - intersection
    
    return intersection / union if union > 0 else 0

def calculate_metrics(pred_boxes: List[List[float]],
                     pred_labels: List[int],
                     pred_scores: List[float],
                     true_boxes: List[List[float]],
                     true_labels: List[int],
                     iou_threshold: float = 0.5) -> Dict[str, float]:
    """Calculate detection metrics (precision, recall, F1-score)."""
    true_positives = 0
    false_positives = 0
    false_negatives = len(true_boxes)
    
    matched_gt = set()
    
    for pred_box, pred_label, pred_score in zip(pred_boxes, pred_labels, pred_scores):
        best_iou = 0
        best_gt_idx = -1
        
        for i, (true_box, true_label) in enumerate(zip(true_boxes, true_labels)):
            if i in matched_gt or pred_label != true_label:
                continue
                
            iou = compute_iou(pred_box, true_box)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = i
        
        if best_iou >= iou_threshold:
            true_positives += 1
            matched_gt.add(best_gt_idx)
            false_negatives -= 1
        else:
            false_positives += 1
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score
    }

def save_model(model: torch.nn.Module, save_path: str):
    """Save model weights."""
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), save_path)

def load_model(model: torch.nn.Module, weights_path: str) -> torch.nn.Module:
    """Load model weights."""
    model.load_state_dict(torch.load(weights_path))
    return model 