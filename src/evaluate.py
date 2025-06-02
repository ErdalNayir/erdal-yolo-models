import argparse
import yaml
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, List
import cv2

from yolov8_model import YOLOv8Detector
from yolov9_model import YOLOv9Detector
from utils_folder import setup_logging, visualize_detection, calculate_metrics

logger = setup_logging(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate and compare YOLO models")
    parser.add_argument("--data", type=str, required=True,
                       help="Path to data configuration file")
    parser.add_argument("--weights-dir", type=str, required=True,
                       help="Directory containing model weights")
    parser.add_argument("--output-dir", type=str, default="comparison_results",
                       help="Directory to save comparison results")
    parser.add_argument("--batch-size", type=int, default=16,
                       help="Batch size for evaluation")
    parser.add_argument("--img-size", type=int, default=640,
                       help="Input image size")
    parser.add_argument("--num-samples", type=int, default=100,
                       help="Number of test samples for visualization")
    return parser.parse_args()

def load_models(weights_dir: str) -> Dict[str, object]:
    """Load YOLOv8 and YOLOv9 models."""
    models = {}
    weights_dir = Path(weights_dir)
    
    # Load YOLOv8
    if (weights_dir / "yolov8").exists():
        models["YOLOv8"] = YOLOv8Detector(
            model_type="yolov8s",
            pretrained=False
        )
    
    # Load YOLOv9
    if (weights_dir / "yolov9").exists():
        models["YOLOv9"] = YOLOv9Detector(
            model_type="yolov9s",
            pretrained=False
        )
    
    return models

def evaluate_models(models: Dict[str, object], args) -> Dict[str, Dict]:
    """Evaluate all models and collect metrics."""
    results = {}
    
    for name, model in models.items():
        logger.info(f"Evaluating {name}")
        eval_results = model.evaluate(
            data_yaml=args.data,
            batch_size=args.batch_size,
            image_size=args.img_size
        )
        results[name] = eval_results
    
    return results

def compare_metrics(results: Dict[str, Dict], output_dir: str):
    """Compare and visualize metrics between models."""
    metrics_df = pd.DataFrame(results).T
    
    # Create comparison plots
    plt.figure(figsize=(15, 10))
    
    # Bar plot for mAP metrics
    plt.subplot(2, 2, 1)
    metrics_df[["mAP50", "mAP50-95"]].plot(kind="bar")
    plt.title("mAP Comparison")
    plt.ylabel("Score")
    plt.xticks(rotation=45)
    
    # Bar plot for precision and recall
    plt.subplot(2, 2, 2)
    metrics_df[["precision", "recall"]].plot(kind="bar")
    plt.title("Precision-Recall Comparison")
    plt.ylabel("Score")
    plt.xticks(rotation=45)
    
    # Save plot
    plt.tight_layout()
    plt.savefig(f"{output_dir}/metrics_comparison.png")
    plt.close()
    
    # Save detailed metrics to CSV
    metrics_df.to_csv(f"{output_dir}/detailed_metrics.csv")

def visualize_comparisons(models: Dict[str, object],
                         data_config: Dict,
                         num_samples: int,
                         output_dir: str):
    """Create visual comparisons of model predictions."""
    # Load test images
    test_dir = Path(data_config["test"])
    image_files = list(test_dir.glob("*.jpg"))[:num_samples]
    
    for i, image_file in enumerate(image_files):
        # Load and preprocess image
        image = cv2.imread(str(image_file))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Create subplot for each model
        plt.figure(figsize=(20, 5))
        plt.subplot(1, len(models) + 1, 1)
        plt.imshow(image)
        plt.title("Original Image")
        plt.axis("off")
        
        # Get predictions from each model
        for j, (name, model) in enumerate(models.items(), 2):
            boxes, labels, scores = model.predict(image)
            
            # Visualize predictions
            vis_img = visualize_detection(
                image, boxes, labels, scores,
                class_names=data_config["names"]
            )
            
            plt.subplot(1, len(models) + 1, j)
            plt.imshow(vis_img)
            plt.title(f"{name} Predictions")
            plt.axis("off")
        
        # Save comparison
        plt.tight_layout()
        plt.savefig(f"{output_dir}/comparison_{i+1}.png")
        plt.close()

def main():
    """Main function."""
    args = parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data configuration
    with open(args.data) as f:
        data_config = yaml.safe_load(f)
    
    # Load models
    models = load_models(args.weights_dir)
    if not models:
        logger.error("No model weights found!")
        return
    
    # Evaluate models
    results = evaluate_models(models, args)
    
    # Compare metrics
    compare_metrics(results, str(output_dir))
    
    # Visualize comparisons
    visualize_comparisons(
        models,
        data_config,
        args.num_samples,
        str(output_dir)
    )
    
    logger.info(f"Comparison results saved to {output_dir}")

if __name__ == "__main__":
    main() 