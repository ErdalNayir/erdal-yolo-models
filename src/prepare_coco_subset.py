import os
import glob
from pathlib import Path
import shutil
import zipfile
import requests
from tqdm import tqdm
import json
import random

def download_file(url: str, save_path: str):
    """Download a file with progress bar."""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(save_path, 'wb') as file, tqdm(
        desc=os.path.basename(save_path),
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for data in response.iter_content(chunk_size=1024):
            size = file.write(data)
            pbar.update(size)

def prepare_coco_subset(base_path: str = "data/coco", num_images: int = 500):
    """
    Download and prepare a subset of COCO dataset.
    Args:
        base_path: Base directory to store the dataset
        num_images: Number of images to include in the subset
    """
    # Create directories
    base_path = Path(base_path)
    base_path.mkdir(parents=True, exist_ok=True)
    
    # Only download validation set and annotations
    urls = {
        'val2017.zip': 'http://images.cocodataset.org/zips/val2017.zip',
        'annotations_trainval2017.zip': 'http://images.cocodataset.org/annotations/annotations_trainval2017.zip'
    }
    
    # Download and extract files
    for filename, url in urls.items():
        zip_path = base_path / filename
        
        # Download if not exists
        if not zip_path.exists():
            print(f"Downloading {filename}...")
            download_file(url, str(zip_path))
        
        # Extract
        print(f"Extracting {filename}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(str(base_path))
        
        # Remove zip file
        zip_path.unlink()
    
    # Create subset directories
    subset_dir = base_path / 'subset'
    (subset_dir / 'train').mkdir(parents=True, exist_ok=True)
    (subset_dir / 'val').mkdir(parents=True, exist_ok=True)
    (subset_dir / 'test').mkdir(parents=True, exist_ok=True)
    
    # Load COCO annotations
    print("Creating dataset subset...")
    with open(base_path / 'annotations' / 'instances_val2017.json', 'r') as f:
        coco = json.load(f)
    
    # Randomly select images
    all_images = coco['images']
    selected_images = random.sample(all_images, num_images)
    
    # Split into train/val/test (70/15/15 split)
    train_split = int(0.7 * num_images)
    val_split = int(0.85 * num_images)
    
    train_images = selected_images[:train_split]          # 70% for training
    val_images = selected_images[train_split:val_split]   # 15% for validation
    test_images = selected_images[val_split:]             # 15% for testing
    
    # Create image lists and copy images
    splits = {
        'train': train_images,
        'val': val_images,
        'test': test_images
    }
    
    # Get selected image IDs
    selected_image_ids = set(img['id'] for img in selected_images)
    
    # Filter annotations for selected images
    selected_annotations = [
        ann for ann in coco['annotations']
        if ann['image_id'] in selected_image_ids
    ]
    
    # Create new annotation files
    for split_name, images in splits.items():
        # Copy images
        print(f"Copying {split_name} images...")
        image_paths = []
        for img in tqdm(images):
            src = base_path / 'val2017' / img['file_name']
            dst = subset_dir / split_name / img['file_name']
            shutil.copy2(src, dst)
            image_paths.append(str(dst))
        
        # Create image list file
        with open(subset_dir / f'{split_name}.txt', 'w') as f:
            for path in image_paths:
                f.write(f'{path}\n')
        
        # Create annotation file
        split_annotations = {
            'info': coco['info'],
            'licenses': coco['licenses'],
            'categories': coco['categories'],
            'images': images,
            'annotations': [
                ann for ann in selected_annotations
                if ann['image_id'] in {img['id'] for img in images}
            ]
        }
        
        with open(subset_dir / f'instances_{split_name}.json', 'w') as f:
            json.dump(split_annotations, f)
    
    # Update COCO dataset config for the subset
    coco_yaml = """
# COCO 2017 subset config for YOLO
# {num_images} images split into train/val/test (70/15/15 split)

path: {base_path}/subset  # dataset root dir
train: train.txt  # train images
val: val.txt      # val images
test: test.txt    # test images

# Classes
nc: 80  # number of classes
names: ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
        'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
        'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
        'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
        'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
        'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
        'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
        'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
        'hair drier', 'toothbrush']
    """.format(base_path=str(base_path), num_images=num_images)
    
    with open(base_path.parent / 'coco.yaml', 'w') as f:
        f.write(coco_yaml.strip())
    
    print(f"""
Dataset preparation completed!
Created subset with {num_images} images:
- Training: {len(train_images)} images ({len(train_images)/num_images*100:.1f}%)
- Validation: {len(val_images)} images ({len(val_images)/num_images*100:.1f}%)
- Testing: {len(test_images)} images ({len(test_images)/num_images*100:.1f}%)
Dataset config saved to {base_path.parent}/coco.yaml
""")

if __name__ == "__main__":
    prepare_coco_subset(num_images=500)  # Creates a subset with 500 images 