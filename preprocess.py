import os
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm
import json
import matplotlib.pyplot as plt


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)
    
    
    
DATA_DIR = Path('brisc2025')
PROCESSED_DIR = Path('data/processed')

CONFIG = {
    'target_size': 224,
    'normalize': True,
    'apply_clahe': True,
    'clahe_clip_limit': 4.0,
    'clahe_tile_size': (8, 8),
    'denoise': True,
    'adaptive_threshold_masks': False,
}

def preprocess_image(img_path, target_size=224, apply_clahe=True, normalize=True, 
                     clahe_clip_limit=2.0, clahe_tile_size=(8, 8), denoise=False):
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if img is None: 
        print(f"Warning: Could not read {img_path}")
        return None
    
    # Validate image dimensions
    if img.shape[0] < 50 or img.shape[1] < 50:
        print(f"Warning: Image {img_path.name} too small ({img.shape}), skipping")
        return None
    
    # Denoise before resizing
    if denoise:
        img = cv2.fastNlMeansDenoising(img, None, h=10, templateWindowSize=7, searchWindowSize=21)
    
    # Resize
    img = cv2.resize(img, (target_size, target_size), interpolation=cv2.INTER_AREA)
    
    # Apply CLAHE
    if apply_clahe:
        clahe = cv2.createCLAHE(clipLimit=clahe_clip_limit, tileGridSize=clahe_tile_size)
        img = clahe.apply(img)
    
    # Normalize to [0, 1]
    if normalize:
        img = img.astype(np.float32) / 255.0
    
    return img

def preprocess_mask(mask_path, target_size=224, adaptive_threshold=False):
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None: 
        print(f"Could not read mask {mask_path}")
        return None
    
    # Resize with NEAREST to preserve binary values
    mask = cv2.resize(mask, (target_size, target_size), interpolation=cv2.INTER_NEAREST)
    

    if adaptive_threshold:
        mask = cv2.adaptiveThreshold(mask, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        mask = (mask > 127).astype(np.float32)
    else:
        mask = (mask > 127).astype(np.float32) 
    
    return mask

def run_classification():
    train_source = DATA_DIR / 'classification_task' / 'train'
    if not train_source.exists():
        print(f"Could not find {train_source}")
        return None

    classes = [d.name for d in train_source.iterdir() if d.is_dir()]
    print(f"Found classes: {classes}")

    class_to_idx = {cls_name: idx for idx, cls_name in enumerate(sorted(classes))}
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    with open(PROCESSED_DIR / 'class_mapping.json', 'w') as f:
        json.dump(class_to_idx, f)

    stats = {
        'total': 0, 
        'processed': 0, 
        'failed': 0, 
        'per_class': {cls: 0 for cls in classes},
        'splits': {'train': 0, 'test': 0}  # Track per-split counts
    }
    
    for split in ['train', 'test']:
        split_dir = DATA_DIR / 'classification_task' / split
        output_base = PROCESSED_DIR / split

        if not split_dir.exists(): 
            print("directory not found")
            continue

        for cls in classes:
            cls_dir = split_dir / cls
            if not cls_dir.exists(): 
                continue

            (output_base / cls).mkdir(parents=True, exist_ok=True)
            
            files = list(cls_dir.glob('*.jpg')) + list(cls_dir.glob('*.jpeg')) + list(cls_dir.glob('*.png'))
            stats['total'] += len(files)
            stats['per_class'][cls] += len(files)
            stats['splits'][split] += len(files)

            for img_path in tqdm(files, desc=f"{split}/{cls}"):
                img = preprocess_image(
                    img_path, 
                    target_size=CONFIG['target_size'],
                    apply_clahe=CONFIG['apply_clahe'],
                    normalize=CONFIG['normalize'],
                    clahe_clip_limit=CONFIG['clahe_clip_limit'],
                    clahe_tile_size=CONFIG['clahe_tile_size'],
                    denoise=CONFIG['denoise']
                )
                if img is not None:
                    np.save(output_base / cls / (img_path.stem + '.npy'), img)
                    stats['processed'] += 1
                else:
                    stats['failed'] += 1
    
    print(f"\nClassification Stats: {stats['processed']}/{stats['total']} processed, {stats['failed']} failed")
    print(f"   Per-class counts: {stats['per_class']}")
    print(f"   Train/Test split: {stats['splits']}")
    
    if stats['per_class']:
        plt.figure(figsize=(10, 5))
        colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12']
        bars = plt.bar(stats['per_class'].keys(), stats['per_class'].values(), 
                       color=colors, edgecolor='black', linewidth=1.5)
        plt.title('Class Distribution (All Splits)', fontsize=14, fontweight='bold')
        plt.xlabel('Tumor Type', fontsize=12)
        plt.ylabel('Number of Samples', fontsize=12)
        plt.grid(axis='y', alpha=0.3)
        
        # Add count labels
        for bar, (cls, count) in zip(bars, stats['per_class'].items()):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{count}', ha='center', va='bottom', fontweight='bold', fontsize=11)
        
        plt.tight_layout()
        plt.savefig(PROCESSED_DIR / 'class_balance.png', dpi=300)
        plt.close()
        print(f"Saved class distribution chart")
    
    return stats

def run_segmentation():
    stats = {
        'total': 0, 
        'processed': 0, 
        'failed': 0, 
        'missing_mask': 0, 
        'fg_pixels': 0, 
        'total_pixels': 0,
        'splits': {'train': 0, 'test': 0}
    }
    
    for split in ['train', 'test']:
        img_dir = DATA_DIR / 'segmentation_task' / split / 'images'
        mask_dir = DATA_DIR / 'segmentation_task' / split / 'masks'

        out_img_dir = PROCESSED_DIR / 'segmentation' / split / 'images'
        out_mask_dir = PROCESSED_DIR / 'segmentation' / split / 'masks'

        out_img_dir.mkdir(parents=True, exist_ok=True)
        out_mask_dir.mkdir(parents=True, exist_ok=True)

        if not img_dir.exists(): 
            print(f"Skipping {split} - directory not found")
            continue

        files = list(img_dir.glob('*.jpg')) + list(img_dir.glob('*.jpeg')) + list(img_dir.glob('*.png'))
        stats['total'] += len(files)
        print(f"Found {len(files)} images in {split}")

        for img_path in tqdm(files, desc=f"Segmentation {split}"):
            mask_path = None
            for ext in ['.png', '.jpg', '.jpeg']:
                potential_mask = mask_dir / (img_path.stem + ext)
                if potential_mask.exists():
                    mask_path = potential_mask
                    break
            
            if mask_path is None:
                print(f"Warning: No mask found for {img_path.name}")
                stats['missing_mask'] += 1
                continue
            
            img = preprocess_image(
                img_path,
                target_size=CONFIG['target_size'],
                apply_clahe=CONFIG['apply_clahe'],
                normalize=CONFIG['normalize'],
                clahe_clip_limit=CONFIG['clahe_clip_limit'],
                clahe_tile_size=CONFIG['clahe_tile_size'],
                denoise=CONFIG['denoise']
            )
            mask = preprocess_mask(mask_path, target_size=CONFIG['target_size'], 
                                   adaptive_threshold=CONFIG['adaptive_threshold_masks'])
            
            if img is not None and mask is not None:
                np.save(out_img_dir / (img_path.stem + '.npy'), img)
                np.save(out_mask_dir / (img_path.stem + '.npy'), mask)
                stats['processed'] += 1
                stats['splits'][split] += 1
                stats['fg_pixels'] += np.sum(mask > 0)
                stats['total_pixels'] += mask.size
            else:
                stats['failed'] += 1
    
    print(f"\nSegmentation Stats: {stats['processed']}/{stats['total']} processed, "
          f"{stats['failed']} failed, {stats['missing_mask']} missing masks")
    print(f"   Train/Test split: {stats['splits']}")
    

    if stats['total_pixels'] > 0:
        fg_ratio = stats['fg_pixels'] / stats['total_pixels']
        pos_weight_suggest = (stats['total_pixels'] - stats['fg_pixels']) / stats['fg_pixels']
        stats['foreground_ratio'] = fg_ratio
        stats['pos_weight_suggestion'] = pos_weight_suggest
        print(f"Foreground pixel ratio: {fg_ratio:.4f}")
        print(f"Suggested pos_weight for BCE: {pos_weight_suggest:.2f}")
    
    return stats

def validate_preprocessing():
    
    validation_results = {'status': 'success', 'checks': []}
    
    # Check classification
    class_mapping_path = PROCESSED_DIR / 'class_mapping.json'
    if class_mapping_path.exists():
        with open(class_mapping_path) as f:
            class_map = json.load(f)
        print(f"Class mapping: {class_map}")
        validation_results['checks'].append(('class_mapping', 'pass', class_map))
    else:
        print("Class mapping not found")
        validation_results['checks'].append(('class_mapping', 'fail', None))
        validation_results['status'] = 'failed'
    
    # Check a sample image
    sample_dirs = list((PROCESSED_DIR / 'train').glob('*/*.npy'))
    if sample_dirs:
        sample = np.load(sample_dirs[0])
        print(f"Sample classification image shape: {sample.shape}")
        print(f"Value range: [{sample.min():.3f}, {sample.max():.3f}]")
        validation_results['checks'].append(('sample_image', 'pass', {
            'shape': sample.shape,
            'min': float(sample.min()),
            'max': float(sample.max())
        }))
    else:
        print("No classification samples found")
        validation_results['checks'].append(('sample_image', 'warn', None))
    
    # Check segmentation
    seg_imgs = list((PROCESSED_DIR / 'segmentation' / 'train' / 'images').glob('*.npy'))
    seg_masks = list((PROCESSED_DIR / 'segmentation' / 'train' / 'masks').glob('*.npy'))
    print(f"Segmentation images: {len(seg_imgs)}, masks: {len(seg_masks)}")
    
    if seg_imgs and seg_masks:
        img = np.load(seg_imgs[0])
        mask = np.load(seg_masks[0])
        print(f"   Image shape: {img.shape}, range: [{img.min():.3f}, {img.max():.3f}]")
        print(f"   Mask shape: {mask.shape}, unique values: {np.unique(mask)}, foreground ratio: {np.mean(mask):.4f}")
        validation_results['checks'].append(('segmentation', 'pass', {
            'num_images': len(seg_imgs),
            'num_masks': len(seg_masks),
            'image_shape': img.shape,
            'mask_unique_values': np.unique(mask).tolist(),
            'mask_foreground_ratio': float(np.mean(mask))
        }))
    else:
        print("No segmentation samples found")
        validation_results['checks'].append(('segmentation', 'warn', None))
    
    return validation_results

if __name__ == '__main__':
    clf_stats = run_classification()
    seg_stats = run_segmentation()
    val_results = validate_preprocessing()
    

    print("Preprocessing Complete!")

    preprocessing_stats = {
        'config': CONFIG,
        'classification': clf_stats if clf_stats else {},
        'segmentation': seg_stats if seg_stats else {},
        'validation': val_results
    }
    
    stats_path = PROCESSED_DIR / 'preprocessing_stats.json'
    with open(stats_path, 'w') as f:
        json.dump(preprocessing_stats, f, indent=2, cls=NpEncoder)
    
    print(f"\nComplete preprocessing statistics saved to:")
    print(f"   {stats_path}")
    print(f"\nVisualization saved to:")
    print(f"{PROCESSED_DIR / 'class_balance.png'}")
    

    if clf_stats:
        print(f"\nSUMMARY:")
        print(f"   Classification: {clf_stats['processed']}/{clf_stats['total']} images")
        print(f"   Segmentation: {seg_stats['processed']}/{seg_stats['total']} images")