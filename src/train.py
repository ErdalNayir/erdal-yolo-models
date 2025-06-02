import argparse
import yaml
from pathlib import Path
import time
from typing import Dict, Any



from yolov8_model import YOLOv8Detector
from yolov9_model import YOLOv9Detector
from utils_codes import setup_logging, plot_training_results

logger = setup_logging(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train YOLO models")
    parser.add_argument("--model", type=str, required=True, choices=["yolov8", "yolov9"],
                       help="YOLO model version to use")
    parser.add_argument("--data", type=str, required=True,
                       help="Path to data configuration file")
    parser.add_argument("--epochs", type=int, default=100,
                       help="Number of epochs to train")
    parser.add_argument("--batch-size", type=int, default=16,
                       help="Batch size")
    parser.add_argument("--img-size", type=int, default=640,
                       help="Input image size")
    parser.add_argument("--workers", type=int, default=4,
                       help="Number of worker threads")
    parser.add_argument("--device", type=str, default="cuda:0",
                    help="Device to use (cuda:0, cuda:1, or cpu)")
    return parser.parse_args()


def load_data_config(config_path: str) -> Dict[str, Any]:
    """Load and validate data configuration file."""
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    required_keys = ["train", "val", "test", "nc", "names"]
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required key '{key}' in data config")
    
    return config

def train_model(args):
    """Train the selected YOLO model."""
    # Load data configuration
    data_config = load_data_config(args.data)
    logger.info(f"Loaded data config with {data_config['nc']} classes")
    
    # Initialize model
    if args.model == "yolov8":
        model = YOLOv8Detector()
    else:  # yolov9
        model = YOLOv9Detector()
    
    # Create results directory
    results_dir = Path("results") / args.model / time.strftime("%Y%m%d_%H%M%S")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Train model
    logger.info(f"Starting {args.model.upper()} training for {args.epochs} epochs")
    results = model.train(
        data_yaml=args.data,
        epochs=args.epochs,
        batch_size=args.batch_size,
        image_size=args.img_size,
        workers=args.workers
    )
    
    # Save training results plot
    if isinstance(results, dict) and "metrics" in results:
        plot_training_results(
            results["metrics"],
            save_path=str(results_dir / "training_results.png")
        )
    
    # Export model
    model.export(format="onnx", save_dir=str(results_dir))
    
    # Evaluate model
    logger.info("Starting model evaluation")
    eval_results = model.evaluate(
        data_yaml=args.data,
        batch_size=args.batch_size,
        image_size=args.img_size
    )
    
    # Save evaluation results
    eval_path = results_dir / "evaluation_results.yaml"
    with open(eval_path, "w") as f:
        yaml.dump(eval_results, f)
    
    logger.info(f"Training completed. Results saved to {results_dir}")

def main():
    """Main function."""
    args = parse_args()
    train_model(args)

if __name__ == "__main__":
    main() 