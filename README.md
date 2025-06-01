# YOLO Object Detection Comparison Project

This project implements and compares two different versions of YOLO (You Only Look Once) for object detection tasks. The implementation includes YOLOv8 and YOLOv9 models, training on a custom dataset, and performance comparison.

## Project Structure

```
├── data/                  # Dataset directory
├── models/               # Trained model weights
├── src/
│   ├── yolov8_model.py   # YOLOv8 implementation
│   ├── yolov9_model.py   # YOLOv9 implementation
│   ├── train.py          # Training script
│   ├── evaluate.py       # Evaluation script
│   ├── utils.py          # Utility functions
│   └── config.py         # Configuration parameters
└── requirements.txt      # Project dependencies
```

## Setup

1. Create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

1. Prepare your dataset in the COCO format and place it in the `data/` directory
2. Train the models:

```bash
python src/train.py --model yolov8  # For YOLOv8
python src/train.py --model yolov9  # For YOLOv9
```

3. Evaluate and compare models:

```bash
python src/evaluate.py
```

## Features

- Implementation of YOLOv8 and YOLOv9
- Training on custom datasets
- Model performance comparison
- Error analysis and visualization
- Model interpretation tools

## Results

The comparison results between YOLOv8 and YOLOv9 will be saved in the results directory, including:

- Accuracy metrics
- Detection speed
- Error analysis
- Visualization of results
