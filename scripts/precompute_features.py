"""Pre-compute ML features for Phase 3 Hybrid Model.

This script extracts HSV, LBP, and GLCM features for all images in the 
training and validation sets and saves them to the feature cache. 
"""

import sys
import hashlib
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import numpy as np
import cv2

# Add project root to path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from src.skin_analysis.phase3.config import Phase3Config
from src.skin_analysis.phase1.features import extract_features

def get_cache_path(img_path: Path, cache_dir: Path) -> Path:
    """Standalone cache path generator (matches Dataset logic)."""
    path_hash = hashlib.md5(str(img_path).encode()).hexdigest()
    return cache_dir / f"{path_hash}.npy"

def process_single_image(args):
    img_path, cache_dir = args
    cache_path = get_cache_path(img_path, cache_dir)
    
    if cache_path.exists():
        return True # Skip if already exists

    try:
        # Extract features from raw image
        image_bgr = cv2.imread(str(img_path))
        if image_bgr is None:
            return False

        # Resize to 224x224 for consistent feature extraction
        image_bgr = cv2.resize(image_bgr, (224, 224), interpolation=cv2.INTER_AREA)
        features = extract_features(image_bgr)

        # Cache features
        np.save(cache_path, features)
        return True
    except Exception as e:
        # print(f"Error processing {img_path}: {e}")
        return False

def main():
    cfg = Phase3Config()
    print(f"🚀 Starting pre-computation for: {cfg.data_dir}")
    
    for split in ["train", "val"]:
        split_dir = cfg.data_dir / split
        cache_dir = cfg.feature_cache_dir / split
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        if not split_dir.exists():
            print(f"⚠️  {split} directory not found: {split_dir}")
            continue
            
        print(f"\n📦 Processing {split} set...")
        
        # Collect all image paths (ignoring class folders for now)
        all_images = []
        for ext in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]:
            all_images.extend(list(split_dir.rglob(f"*{ext}")))
            all_images.extend(list(split_dir.rglob(f"*{ext.upper()}")))
            
        if not all_images:
            print(f"❌ No images found in {split_dir}")
            continue

        print(f"Found {len(all_images)} images. Using all available CPU cores...")
        
        # Parallel processing
        tasks = [(img, cache_dir) for img in all_images]
        with ProcessPoolExecutor() as executor:
            list(tqdm(executor.map(process_single_image, tasks), total=len(tasks)))
            
    print("\n✅ Pre-computation complete! Start training with 'make train-c'.")

if __name__ == "__main__":
    main()
