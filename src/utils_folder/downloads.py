import os
import requests
from pathlib import Path
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)

def download(url: str, save_dir: str = None) -> str:
    """
    Download a file from a URL with progress bar.
    
    Args:
        url (str): URL to download from
        save_dir (str, optional): Directory to save the file. Defaults to current directory.
    
    Returns:
        str: Path to downloaded file
    """
    # Get filename from URL
    filename = url.split('/')[-1]
    
    # Set save directory
    if save_dir:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        save_path = Path(save_dir) / filename
    else:
        save_path = Path(filename)
    
    # Don't download if file exists
    if save_path.exists():
        logger.info(f"File {filename} already exists, skipping download")
        return str(save_path)
    
    # Download with progress bar
    logger.info(f"Downloading {filename} from {url}")
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(save_path, 'wb') as file, tqdm(
        desc=filename,
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for data in response.iter_content(chunk_size=1024):
            size = file.write(data)
            pbar.update(size)
    
    return str(save_path)

def download_coco_dataset(save_dir: str = 'data/coco'):
    """
    Download COCO dataset.
    
    Args:
        save_dir (str): Directory to save the dataset
    """
    urls = [
        'http://images.cocodataset.org/zips/train2017.zip',
        'http://images.cocodataset.org/zips/val2017.zip',
        'http://images.cocodataset.org/zips/test2017.zip',
        'http://images.cocodataset.org/annotations/annotations_trainval2017.zip'
    ]
    
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    for url in urls:
        # Download file
        zip_path = download(url, str(save_dir))
        
        # Extract zip file
        import zipfile
        logger.info(f"Extracting {zip_path}")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(str(save_dir))
        
        # Remove zip file
        os.remove(zip_path)
        logger.info(f"Removed {zip_path}")

if __name__ == "__main__":
    # Setup basic logging
    logging.basicConfig(level=logging.INFO)
    
    # Example usage
    download_coco_dataset() 