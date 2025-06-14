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
from utils_codes import setup_logging, visualize_detection, calculate_metrics

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
            model_type=f"{weights_dir}/yolov8/best.pt",
            pretrained=False
        )
    
    # Load YOLOv9
    if (weights_dir / "yolov9").exists():
        models["YOLOv9"] = YOLOv9Detector(
            model_type=f"{weights_dir}/yolov9/best.pt",
            pretrained=False
        )
    
    return models

def evaluate_models(models: Dict[str, object], args) -> Dict[str, Dict]:
    """Evaluate all models and collect metrics."""
    results = {}
    
    for name, model in models.items():
        logger.info(f"Evaluating {name}")
        try:
            eval_results = model.evaluate(
                data_yaml=args.data,
                batch_size=args.batch_size,
                image_size=args.img_size
            )
            
            # Extract and format metrics properly
            formatted_results = extract_yolo_metrics(eval_results)
            results[name] = formatted_results
            
            logger.info(f"{name} evaluation completed successfully")
            
        except Exception as e:
            logger.error(f"Error evaluating {name}: {e}")
            results[name] = {"error": str(e)}
    
    return results

def compare_metrics(results: Dict[str, Dict], output_dir: str):
    """Compare and visualize metrics between models."""
    
    # Debug: Print the structure of results to understand the issue
    print("Results structure:")
    for model_name, model_results in results.items():
        print(f"{model_name}: {type(model_results)} - {model_results}")
    
    # Handle different result structures
    try:
        # Try the original approach first
        if all(isinstance(v, dict) for v in results.values()):
            # If results is properly structured as {model: {metric: value}}
            metrics_df = pd.DataFrame(results).T
        else:
            # If results contains scalar values or other structures
            # Convert to proper format
            formatted_results = {}
            for model_name, model_results in results.items():
                if isinstance(model_results, dict):
                    formatted_results[model_name] = model_results
                else:
                    # If model_results is not a dict, wrap it or extract metrics
                    # This depends on what your model.evaluate() actually returns
                    formatted_results[model_name] = {"overall_metric": model_results}
            
            metrics_df = pd.DataFrame(formatted_results).T
    
    except ValueError as e:
        print(f"Error creating DataFrame: {e}")
        # Fallback: create a simple comparison table
        metrics_data = []
        for model_name, model_results in results.items():
            if isinstance(model_results, dict):
                row = {"Model": model_name}
                row.update(model_results)
                metrics_data.append(row)
            else:
                metrics_data.append({"Model": model_name, "Result": str(model_results)})
        
        metrics_df = pd.DataFrame(metrics_data)
        if "Model" in metrics_df.columns:
            metrics_df = metrics_df.set_index("Model")
    
    print("Metrics DataFrame:")
    print(metrics_df)
    
    # Create comparison plots only if we have numeric data
    numeric_columns = metrics_df.select_dtypes(include=[np.number]).columns
    
    if len(numeric_columns) > 0:
        plt.figure(figsize=(15, 10))
        
        # Plot 1: All numeric metrics
        plt.subplot(2, 2, 1)
        if len(numeric_columns) <= 10:  # Avoid overcrowded plots
            metrics_df[numeric_columns].plot(kind="bar", ax=plt.gca())
            plt.title("All Metrics Comparison")
            plt.ylabel("Score")
            plt.xticks(rotation=45)
        
        # Plot 2: mAP metrics if available
        plt.subplot(2, 2, 2)
        map_columns = [col for col in numeric_columns if 'map' in col.lower() or 'ap' in col.lower()]
        if map_columns:
            metrics_df[map_columns].plot(kind="bar", ax=plt.gca())
            plt.title("mAP Metrics Comparison")
            plt.ylabel("Score")
            plt.xticks(rotation=45)
        
        # Plot 3: Precision/Recall if available
        plt.subplot(2, 2, 3)
        pr_columns = [col for col in numeric_columns if any(x in col.lower() for x in ['precision', 'recall', 'p', 'r'])]
        if pr_columns:
            metrics_df[pr_columns].plot(kind="bar", ax=plt.gca())
            plt.title("Precision-Recall Comparison")
            plt.ylabel("Score")
            plt.xticks(rotation=45)
        
        # Plot 4: F1 Score if available
        plt.subplot(2, 2, 4)
        f1_columns = [col for col in numeric_columns if 'f1' in col.lower()]
        if f1_columns:
            metrics_df[f1_columns].plot(kind="bar", ax=plt.gca())
            plt.title("F1 Score Comparison")
            plt.ylabel("Score")
            plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/metrics_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
    else:
        print("No numeric metrics found for plotting")
    
    # Save detailed metrics to CSV
    metrics_df.to_csv(f"{output_dir}/detailed_metrics.csv")
    print(f"Metrics saved to {output_dir}/detailed_metrics.csv")

def extract_yolo_metrics(eval_output):
    """Extract metrics from YOLO evaluation output"""
    metrics = {}
    
    # If eval_output is already a dict, return as is
    if isinstance(eval_output, dict):
        return eval_output
    
    # If eval_output is a string or other format, parse it
    # This is a placeholder - you'll need to adapt based on actual output format
    if hasattr(eval_output, 'results'):
        # For newer YOLO versions
        results = eval_output.results
        if hasattr(results, 'results_dict'):
            return results.results_dict
    
    # Default fallback
    return {"evaluation_result": str(eval_output)}

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