import os
import glob
from pathlib import Path
import shutil
import zipfile
import requests
from tqdm import tqdm

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

def prepare_coco_dataset(base_path: str = "data/coco"):
    """
    Download and prepare COCO dataset.
    Args:
        base_path: Base directory to store the dataset
    """
    # Create directories
    base_path = Path(base_path)
    base_path.mkdir(parents=True, exist_ok=True)
    
    # COCO dataset URLs
    urls = {
        'train2017.zip': 'http://images.cocodataset.org/zips/train2017.zip',
        'val2017.zip': 'http://images.cocodataset.org/zips/val2017.zip',
        'test2017.zip': 'http://images.cocodataset.org/zips/test2017.zip',
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
    
    # Create image lists
    print("Creating image lists...")
    for split in ['train2017', 'val2017', 'test2017']:
        img_dir = base_path / split
        if img_dir.exists():
            # Create image list file
            with open(base_path / f'{split}.txt', 'w') as f:
                img_files = sorted(glob.glob(str(img_dir / '*.jpg')))
                for img_file in img_files:
                    f.write(f'{img_file}\n')

    # Update COCO dataset config
    coco_yaml = """
# COCO 2017 dataset config for YOLO
# Train/val/test sets as 1) dir: path/to/imgs, 2) file: path/to/imgs.txt, or 3) list: [path/to/imgs1, path/to/imgs2, ..]

path: {base_path}  # dataset root dir
train: train2017.txt  # train images
val: val2017.txt  # val images
test: test2017.txt  # test images (optional)

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
    """.format(base_path=str(base_path))
    
    with open(base_path.parent / 'coco.yaml', 'w') as f:
        f.write(coco_yaml.strip())
    
    print("COCO dataset preparation completed!")

if __name__ == "__main__":
    prepare_coco_dataset() 