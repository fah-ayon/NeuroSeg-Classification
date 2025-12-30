import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import json
import argparse
import sys
from models import UNetWithClassifier, AttentionUNet, UNet, UNetClassifier


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
CLASS_MAPPING_PATH = 'data/processed/class_mapping.json'
TARGET_SIZE = 224

MODEL_CONFIGS = {
    'joint': {
        'path': 'models/unet_joint_best.pth',
        'type': 'joint',
        'name': 'Joint Model'
    },
    'attention': {
        'path': 'models/att_unet_best.pth',
        'type': 'attention',
        'name': 'Attention U-Net'
    },
    'unet': {
        'path': 'models/unet_seg_best.pth',
        'type': 'unet',
        'name': 'Base U-Net'
    },
    'classifier': {
        'path': 'models/unet_clf_best.pth',
        'type': 'classifier',
        'name': 'Classifier'
    }
}

def load_model(model_path, model_type):
    print(f"Loading {model_type} architecture on {DEVICE}...")
    
    if model_type == 'joint':
        model = UNetWithClassifier(n_channels=1, n_seg_classes=1, n_clf_classes=4)
    elif model_type == 'attention':
        model = AttentionUNet(n_channels=1, n_classes=1)
    elif model_type == 'unet':
        model = UNet(n_channels=1, n_classes=1)
    elif model_type == 'classifier':
        model = UNetClassifier(n_channels=1, n_classes=4)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    model = model.to(DEVICE)
    
    try:
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        model.eval()
        print(f"Loaded weights from {model_path}")
        return model
    except Exception as e:
        print(f"Error loading weights: {e}")
        return None

def load_class_mapping():
    try:
        with open(CLASS_MAPPING_PATH, 'r') as f:
            mapping = json.load(f)
        return {v: k for k, v in mapping.items()}
    except:
        return {0: 'glioma', 1: 'meningioma', 2: 'no_tumor', 3: 'pituitary'}

def preprocess_image(img_path):
    if isinstance(img_path, (str, Path)):
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Could not read image: {img_path}")
    else:
        img = img_path
    
    img = cv2.fastNlMeansDenoising(img, None, h=10, templateWindowSize=7, searchWindowSize=21)
    img = cv2.resize(img, (TARGET_SIZE, TARGET_SIZE), interpolation=cv2.INTER_AREA)
    
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    img = clahe.apply(img)
    img = img.astype(np.float32) / 255.0
    return img

def detect_mri_view(img):
    h, w = img.shape
    if h < 10 or w < 10:
        return "Unknown", 0.5
    
    aspect_ratio = w / h
    h_start, h_end = max(0, h//4), min(h, 3*h//4)
    w_start, w_end = max(0, w//4), min(w, 3*w//4)
    
    if h_end <= h_start or w_end <= w_start:
        return "Unknown", 0.5
    
    center_region = img[h_start:h_end, w_start:w_end]
    edge_parts = []
    if h//4 > 0:
        edge_parts.append(img[0:h//4, :].flatten())
    if h - 3*h//4 > 0:
        edge_parts.append(img[3*h//4:, :].flatten())
    if w//4 > 0:
        edge_parts.append(img[:, 0:w//4].flatten())
    if w - 3*w//4 > 0:
        edge_parts.append(img[:, 3*w//4:].flatten())
    
    if not edge_parts:
        edge_region = img.flatten()
    else:
        edge_region = np.concatenate(edge_parts)
    
    center_mean = np.mean(center_region)
    edge_mean = np.mean(edge_region)
    center_std = np.std(center_region)
    mid_w = w // 2
    if mid_w > 0:
        left_half = img[:, :mid_w]
        right_half = np.fliplr(img[:, w-mid_w:])
        if left_half.shape == right_half.shape:
            try:
                horizontal_symmetry = np.corrcoef(left_half.flatten(), right_half.flatten())[0, 1]
                if np.isnan(horizontal_symmetry):
                    horizontal_symmetry = 0.0
            except:
                horizontal_symmetry = 0.0
        else:
            horizontal_symmetry = 0.0
    else:
        horizontal_symmetry = 0.0
        
    mid_h = h // 2
    if mid_h > 0:
        top_half = img[:mid_h, :]
        bottom_half = np.flipud(img[h-mid_h:, :])
        
        if top_half.shape == bottom_half.shape:
            try:
                vertical_symmetry = np.corrcoef(top_half.flatten(), bottom_half.flatten())[0, 1]
                if np.isnan(vertical_symmetry):
                    vertical_symmetry = 0.0
            except:
                vertical_symmetry = 0.0
        else:
            vertical_symmetry = 0.0
    else:
        vertical_symmetry = 0.0
    
    scores = {'Axial': 0, 'Coronal': 0, 'Sagittal': 0}
    

    if horizontal_symmetry > 0.7 and aspect_ratio > 0.85 and aspect_ratio < 1.15:
        scores['Axial'] += 3
    if center_mean > edge_mean * 1.2:
        scores['Axial'] += 2
    
    if horizontal_symmetry > 0.6 and aspect_ratio < 0.9:
        scores['Coronal'] += 3
    if vertical_symmetry < 0.6:
        scores['Coronal'] += 1

    if horizontal_symmetry < 0.5:
        scores['Sagittal'] += 4
    if aspect_ratio > 0.8 and aspect_ratio < 1.2 and center_std > 0.15:
        scores['Sagittal'] += 2
    
    best_view = max(scores, key=scores.get)
    max_score = scores[best_view]
    total_score = sum(scores.values())
    confidence = max_score / max(total_score, 1) if total_score > 0 else 0.33
    
    if confidence < 0.4:
        best_view = "Unknown"
        confidence = 0.5
    
    return best_view, confidence

def predict(model, img_preprocessed, idx_to_class, model_type):
    img_tensor = torch.from_numpy(img_preprocessed).unsqueeze(0).unsqueeze(0).float().to(DEVICE)
    
    pred_class = "N/A"
    confidence = 0.0
    seg_mask = np.zeros((TARGET_SIZE, TARGET_SIZE), dtype=np.float32)
    all_probs = [0.0] * 4

    with torch.no_grad():
        output = model(img_tensor)
        
        if model_type == 'joint':
            seg_logits, clf_logits = output
            seg_mask = torch.sigmoid(seg_logits).squeeze().cpu().numpy()
            probs = torch.softmax(clf_logits, dim=1).squeeze().cpu().numpy()
            pred_idx = np.argmax(probs)
            confidence = probs[pred_idx]
            pred_class = idx_to_class[pred_idx]
            all_probs = probs

        elif model_type in ['attention', 'unet']:
            seg_logits = output
            seg_mask = torch.sigmoid(seg_logits).squeeze().cpu().numpy()
            pred_class = "Segmentation Only"
            confidence = 1.0
        
        elif model_type == 'classifier':
            clf_logits = output
            probs = torch.softmax(clf_logits, dim=1).squeeze().cpu().numpy()
            pred_idx = np.argmax(probs)
            confidence = probs[pred_idx]
            pred_class = idx_to_class[pred_idx]
            all_probs = probs

    seg_mask_binary = (seg_mask > 0.5).astype(np.float32)
    return pred_class, confidence, seg_mask_binary, all_probs

def determine_best_model(all_results):
    best_model = None
    best_score = -1
    
    for model_key, result in all_results.items():
        if result['confidence'] > 0 and result['pred_class'] != "Segmentation Only":
            score = result['confidence']
        elif result['pred_class'] == "Segmentation Only":
            tumor_pct = result['tumor_percentage']
            score = 0.7 if tumor_pct > 0.01 else 0.3
        else:
            score = 0
        
        if score > best_score:
            best_score = score
            best_model = model_key
    
    return best_model

def visualize_all_results(img_original, all_results, idx_to_class, best_model, mri_view, view_conf, save_path=None):
    n_models = len(all_results)
    fig = plt.figure(figsize=(20, 5.5 * n_models))
    
    view_color = 'green' if view_conf > 0.6 else 'orange' if view_conf > 0.4 else 'red'
    view_status = '✓' if view_conf > 0.6 else '⚠' if view_conf > 0.4 else '✗'
    
    fig.suptitle(f'{view_status} MRI View: {mri_view} | Confidence: {view_conf:.1%}', 
                 fontsize=16, fontweight='bold', color=view_color, y=0.995)
    
    view_descriptions = {
        'Axial': 'Axial-Horizontal slice (top-down view)',
        'Coronal': 'Coronal-Frontal slice (front-back view)',
        'Sagittal': 'Sagittal-Side slice (left-right view)',
        'Unknown': 'Unknown-View type could not be determined'
    }
    view_desc = view_descriptions.get(mri_view, '')
    fig.text(0.5, 0.97, view_desc, ha='center', fontsize=11, style='italic', color='gray')
    
    row = 0
    for model_key, result in all_results.items():
        is_best = (model_key == best_model)
        border_color = 'green' if is_best else 'gray'
        title_prefix = '⭐ BEST: ' if is_best else ''
        
        model_name = MODEL_CONFIGS[model_key]['name']
        pred_class = result['pred_class']
        confidence = result['confidence']
        seg_mask = result['seg_mask']
        all_probs = result['all_probs']
        model_type = MODEL_CONFIGS[model_key]['type']

        ax1 = plt.subplot(n_models, 3, row * 3 + 1)
        ax1.imshow(img_original, cmap='gray')
        title_text = f'{title_prefix}{model_name}'
        ax1.set_title(title_text, fontsize=10, fontweight='bold', color=border_color, pad=8)
        ax1.axis('off')
        for spine in ax1.spines.values():
            spine.set_edgecolor(border_color)
            spine.set_linewidth(4 if is_best else 1.5)
        
        ax2 = plt.subplot(n_models, 3, row * 3 + 2)
        if model_type == 'classifier':
            ax2.text(0.5, 0.5, "No Mask Output\n(Classifier Only)", 
                    ha='center', va='center', fontsize=10, color='gray')
            ax2.set_title('Segmentation', fontsize=10, fontweight='bold', pad=8)
            ax2.axis('off')
        else:
            ax2.imshow(seg_mask, cmap='hot')
            tumor_pct = np.mean(seg_mask)
            tumor_status = '🔴' if tumor_pct > 0.05 else '🟡' if tumor_pct > 0.01 else '🟢'
            ax2.set_title(f'{tumor_status} Tumor: {tumor_pct:.2%}', 
                         fontsize=10, fontweight='bold', color=border_color, pad=8)
            ax2.axis('off')
        for spine in ax2.spines.values():
            spine.set_edgecolor(border_color)
            spine.set_linewidth(4 if is_best else 1.5)
        
        ax3 = plt.subplot(n_models, 3, row * 3 + 3)
        if model_type in ['attention', 'unet']:
            ax3.text(0.5, 0.5, "No Class Output\n(Segmentation Only)", 
                    ha='center', va='center', fontsize=10, color='gray')
            ax3.set_title('Classification', fontsize=10, fontweight='bold', pad=8)
            ax3.axis('off')
        else:
            class_names = [idx_to_class[i] for i in sorted(idx_to_class.keys())]
            colors = ['green' if idx_to_class[i] == pred_class else 'lightgray' 
                     for i in sorted(idx_to_class.keys())]
            
            bars = ax3.barh(class_names, all_probs, color=colors, height=0.6, edgecolor='black', linewidth=1)
            ax3.set_xlim(0, 1.05)
            
            conf_indicator = '🟢' if confidence > 0.8 else '🟡' if confidence > 0.6 else '🔴'
            title_color = 'darkgreen' if confidence > 0.8 else 'darkorange' if confidence > 0.6 else 'darkred'
            ax3.set_title(f'{conf_indicator} {pred_class}: {confidence:.1%}', 
                         fontsize=10, fontweight='bold', color=title_color, pad=8)
            ax3.grid(axis='x', alpha=0.3, linestyle='--')
            ax3.tick_params(labelsize=9)
            ax3.set_xlabel('Probability', fontsize=8)
            
            for i, (bar, prob) in enumerate(zip(bars, all_probs)):
                if prob > 0.03:
                    ax3.text(prob + 0.02, i, f'{prob:.1%}', va='center', fontsize=9, fontweight='bold')
        
        for spine in ax3.spines.values():
            spine.set_edgecolor(border_color)
            spine.set_linewidth(4 if is_best else 1.5)
        
        row += 1

    plt.subplots_adjust(left=0.05, right=0.98, top=0.94, bottom=0.03, hspace=0.45, wspace=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved plot to {save_path}")
    
    plt.show()

def run_all_models_inference(image_path, output_folder):
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    
    print(f"Processing image: {image_path}")

    img_original = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img_original is None:
        raise ValueError(f"Could not read image: {image_path}")
    
    print("Detecting MRI view type")
    mri_view, view_conf = detect_mri_view(img_original)
    print(f"   Detected View: {mri_view} (Confidence: {view_conf:.1%})")
    
    img_prep = preprocess_image(image_path)
    idx_to_class = load_class_mapping()
    
    all_results = {}
    
    for model_key, config in MODEL_CONFIGS.items():
        model_path = Path(config['path'])
        
        if not model_path.exists():
            print(f"  {config['name']} not found at {model_path}, skipping...")
            continue
        
        print(f"Running {config['name']}...")
        
        model = load_model(model_path, config['type'])
        if model is None:
            continue
        
        pc, conf, mask, probs = predict(model, img_prep, idx_to_class, config['type'])
        
        all_results[model_key] = {
            'pred_class': pc,
            'confidence': conf,
            'seg_mask': mask,
            'all_probs': probs,
            'tumor_percentage': np.mean(mask)
        }
        
        print(f"Prediction: {pc}")
        if config['type'] not in ['attention', 'unet']:
            print(f"Confidence: {conf:.2%}")
        if config['type'] != 'classifier':
            print(f"Tumor Region: {np.mean(mask):.2%} coverage")
    
    if not all_results:
        print("\nNo models could be loaded. Please check model paths.")
        return
    
    best_model = determine_best_model(all_results)
    
    print(f"BEST MODEL: {MODEL_CONFIGS[best_model]['name']}")

    
    save_path = Path(output_folder) / f"{Path(image_path).stem}_all_models_result.png"
    visualize_all_results(img_original, all_results, idx_to_class, best_model, mri_view, view_conf, save_path)
    
    summary = {
        'image': str(image_path),
        'mri_view': {
            'type': mri_view,
            'confidence': float(view_conf)
        },
        'best_model': {
            'name': MODEL_CONFIGS[best_model]['name'],
            'key': best_model
        },
        'results': {}
    }
    
    for model_key, result in all_results.items():
        summary['results'][model_key] = {
            'model_name': MODEL_CONFIGS[model_key]['name'],
            'prediction': result['pred_class'],
            'confidence': float(result['confidence']),
            'tumor_percentage': float(result['tumor_percentage']),
            'is_best': (model_key == best_model)
        }
    
    json_path = Path(output_folder) / f"{Path(image_path).stem}_summary.json"
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to {json_path}")

def batch_inference(image_folder, output_folder):
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    image_paths = []
    for ext in extensions:
        image_paths.extend(Path(image_folder).glob(ext))
    
    image_paths = list(set([p.resolve() for p in image_paths]))
    image_paths = sorted(image_paths)
    
    if not image_paths:
        print(f"No images found in {image_folder}")
        return

    print(f"Found {len(image_paths)} images. Processing...\n")
    
    success_count = 0
    fail_count = 0
    
    for idx, img_path in enumerate(image_paths, 1):
        print(f"\n{'='*70}")
        print(f"[{idx}/{len(image_paths)}] Processing: {img_path.name}")
        print(f"{'='*70}")
        
        try:
            run_all_models_inference(img_path, output_folder)
            success_count += 1
            print(f"✓ SUCCESS: {img_path.name}")
        except Exception as e:
            fail_count += 1
            print(f"\nFAILED: {img_path.name}")
            print(f"Error: {e}")
            print("\nFull traceback:")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*70}")
    print(f"Batch processing complete!")
    print(f"Success: {success_count}/{len(image_paths)}")
    print(f"Failed: {fail_count}/{len(image_paths)}")
    print(f"Results in {output_folder}")
    print(f"{'='*70}")

def main():
    parser = argparse.ArgumentParser(description='Universal Brain Tumor Inference - All Models')
    
    parser.add_argument('--image', type=str, help='Path to a single image file')
    parser.add_argument('--folder', type=str, help='Path to a folder of images')
    parser.add_argument('--output', type=str, default='results/demo_results', 
                        help='Folder to save results')
    
    args = parser.parse_args()
    
    if args.image:
        run_all_models_inference(args.image, args.output)
    elif args.folder:
        batch_inference(args.folder, args.output)
        
if __name__ == '__main__':
    main()