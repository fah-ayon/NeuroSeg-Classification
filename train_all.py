import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Dataset
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import sys
import os
from models import UNetClassifier, AttentionUNet, UNet, UNetWithClassifier
from dataset import BRISCClassificationDataset, BRISCSegmentationDataset, get_classification_transforms, get_segmentation_transforms


# CONFIGURATION & HYPERPARAMETERS

HYPERPARAMS = {
    "LR_CLF": 1e-4,
    "LR_SEG": 1e-3,
    "BATCH_CLF": 32,
    "BATCH_SEG": 16,
    "BATCH_JOINT": 12,
    "EPOCHS": 100,
    "OPTIMIZER": "AdamW",
    "LOSS_CLF": "CrossEntropyLoss",
    "LOSS_SEG": "BCEWithLogitsLoss (Weighted) + DiceLoss",
    "EARLY_STOP_PATIENCE": 15,
    "WEIGHT_DECAY": 1e-3
}

# Dataset stats from preprocessing
CLASS_COUNTS = {'glioma': 1401, 'meningioma': 1635, 'no_tumor': 1207, 'pituitary': 1757}
total_samples = sum(CLASS_COUNTS.values())
class_weights_list = [total_samples / (4 * CLASS_COUNTS[cls]) for cls in sorted(CLASS_COUNTS.keys())]
POS_WEIGHT = 5.0 #58.0


#LOGGING & PLOTTING

class Logger(object):
    def __init__(self, filename="results/training_log.txt"):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding='utf-8')
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()
    def flush(self):
        self.terminal.flush()

def plot_learning_curves(history, title, filename):
    plt.figure(figsize=(14, 6))
    
    plt.subplot(1, 2, 1)
    if 'train_loss' in history:
        plt.plot(history['train_loss'], label='Train Loss', color='#e74c3c')
    if 'val_loss' in history:
        plt.plot(history['val_loss'], label='Val Loss', color='#3498db')
    plt.title(f'{title} - Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    if 'train_acc' in history:
        plt.plot(history['train_acc'], label='Train Acc', color='#2ecc71', linestyle='--')
    if 'val_acc' in history:
        plt.plot(history['val_acc'], label='Val Acc', color='#9b59b6')
    if 'val_iou' in history: 
        plt.plot(history['val_iou'], label='Val mIoU', color='#f39c12', linewidth=2)
    if 'val_dice' in history: 
        plt.plot(history['val_dice'], label='Val Dice', color='#8e44ad', linestyle='-.')
    if 'val_pixel_acc' in history: 
        plt.plot(history['val_pixel_acc'], label='Val Pixel Acc', color='#16a085', linestyle=':')    

        

    
    plt.title(f'{title} - Metrics')
    plt.xlabel('Epochs')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'results/{filename}', dpi=300)
    plt.close()


#EARLY STOPPING
class EarlyStopping:
    def __init__(self, patience=15, min_delta=0, mode='max'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
            return False
            
        if self.mode == 'max':
            if score < self.best_score + self.min_delta:
                self.counter += 1
            else:
                self.best_score = score
                self.counter = 0
        else:  # mode == 'min'
            if score > self.best_score - self.min_delta:
                self.counter += 1
            else:
                self.best_score = score
                self.counter = 0
        
        if self.counter >= self.patience:
            self.early_stop = True
        return self.early_stop


# METRICS & LOSSES

class DiceLoss(nn.Module):
    def __init__(self, smooth=1):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        
    def forward(self, inputs, targets):
        inputs = torch.sigmoid(inputs).view(-1)
        targets = targets.view(-1)
        intersection = (inputs * targets).sum()
        dice = (2. * intersection + self.smooth) / (inputs.sum() + targets.sum() + self.smooth)
        return 1 - dice

def calculate_seg_metrics(pred_mask, true_mask):
    pred_mask = pred_mask.view(-1)
    true_mask = true_mask.view(-1)
    intersection = (pred_mask * true_mask).sum()
    total = pred_mask.sum() + true_mask.sum()
    union = total - intersection
    dice = (2. * intersection + 1e-6) / (total + 1e-6)
    iou = (intersection + 1e-6) / (union + 1e-6)
    correct = (pred_mask == true_mask).sum()
    pixel_acc = correct / (len(true_mask) + 1e-6)
    return dice.item(), iou.item(), pixel_acc.item()


#JOINT DATASET

class NpyJointDataset(Dataset):
    def __init__(self, data_dir, split='train', transform=None):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.samples = []
        seg_img_dir = self.data_dir / 'segmentation' / split / 'images'
        seg_mask_dir = self.data_dir / 'segmentation' / split / 'masks'
        self.file_to_label = {}
        
        with open(self.data_dir / 'class_mapping.json') as f:
            class_map = json.load(f)
        
        for cls_name, cls_idx in class_map.items():
            cls_path = self.data_dir / split / cls_name
            if cls_path.exists():
                for f in cls_path.glob('*.npy'):
                    self.file_to_label[f.name] = cls_idx
        
        for img_path in seg_img_dir.glob('*.npy'):
            mask_path = seg_mask_dir / img_path.name
            if img_path.name in self.file_to_label and mask_path.exists():
                self.samples.append((img_path, mask_path, self.file_to_label[img_path.name]))

    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        img_path, mask_path, label = self.samples[idx]
        img = np.load(img_path)
        mask = np.load(mask_path)
        img = np.expand_dims(img, axis=-1)
        mask = np.expand_dims(mask, axis=-1)
        
        if self.transform:
            aug = self.transform(image=img, mask=mask)
            img, mask = aug['image'], aug['mask']
            if not isinstance(mask, torch.Tensor):
                mask = torch.from_numpy(mask).permute(2, 0, 1).float()
            elif mask.shape[-1] == 1:
                mask = mask.permute(2, 0, 1).float()
        else:
            img = torch.from_numpy(img).permute(2, 0, 1).float()
            mask = torch.from_numpy(mask).permute(2, 0, 1).float()
        
        return img, mask, torch.tensor(label, dtype=torch.long)


# EVALUATION FUNCTIONS

def evaluate_classification(model, test_loader, criterion, device):
    model.eval()
    test_loss = 0
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for img, label in test_loader:
            img, label = img.to(device), label.to(device)
            out = model(img)
            test_loss += criterion(out, label).item()
            _, pred = torch.max(out, 1)
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(label.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    test_acc = (all_preds == all_labels).mean()
    prec = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    cm = confusion_matrix(all_labels, all_preds)
    
    return {
        'loss': test_loss / len(test_loader),
        'acc': test_acc,
        'precision': prec,
        'recall': recall,
        'f1': f1,
        'cm': cm
    }

def evaluate_segmentation(model, test_loader, bce, dice_loss_fn, device):
    """Evaluate segmentation model on test set"""
    model.eval()
    test_loss = 0
    total_iou, total_dice, total_acc = 0, 0, 0
    
    with torch.no_grad():
        for img, mask in test_loader:
            img, mask = img.to(device), mask.to(device)
            out = model(img)
            test_loss += (bce(out, mask) + dice_loss_fn(out, mask)).item()
            pred = (torch.sigmoid(out) > 0.5).float()
            d, i, a = calculate_seg_metrics(pred, mask)
            total_iou += i
            total_dice += d
            total_acc += a
    
    return {
        'loss': test_loss / len(test_loader),
        'iou': total_iou / len(test_loader),
        'dice': total_dice / len(test_loader),
        'pixel_acc': total_acc / len(test_loader)
    }

def evaluate_joint(model, test_loader, bce, dice_loss_fn, criterion_clf, device):
    model.eval()
    test_loss = 0
    total_iou, total_dice, total_seg_acc = 0, 0, 0
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for img, mask, label in test_loader:
            img, mask, label = img.to(device), mask.to(device), label.to(device)
            seg_out, clf_out = model(img)
            
            seg_loss = bce(seg_out, mask) + dice_loss_fn(seg_out, mask)
            clf_loss = criterion_clf(clf_out, label)
            test_loss += (seg_loss + clf_loss).item()
            
            pred_mask = (torch.sigmoid(seg_out) > 0.5).float()
            d, i, a = calculate_seg_metrics(pred_mask, mask)
            total_iou += i
            total_dice += d
            total_seg_acc += a
            
            _, pred_clf = torch.max(clf_out, 1)
            all_preds.extend(pred_clf.cpu().numpy())
            all_labels.extend(label.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    clf_acc = (all_preds == all_labels).mean()
    
    return {
        'loss': test_loss / len(test_loader),
        'clf_acc': clf_acc,
        'iou': total_iou / len(test_loader),
        'dice': total_dice / len(test_loader),
        'pixel_acc': total_seg_acc / len(test_loader),
        'cm': confusion_matrix(all_labels, all_preds)
    }

def save_confusion_matrix(cm, title, filename):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(f'results/{filename}', dpi=300)
    plt.close()


# MAIN TRAINING PIPELINE

if __name__ == '__main__':
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    class_weights = torch.tensor(class_weights_list).to(DEVICE)
    pos_weight = torch.tensor([POS_WEIGHT]).to(DEVICE)
    
    Path('models').mkdir(exist_ok=True)
    Path('results').mkdir(exist_ok=True)
    sys.stdout = Logger("results/training_log.txt")
    
    print(f"Running on {DEVICE}")
    print("HYPERPARAMETERS SUMMARY")
    for k, v in HYPERPARAMS.items():
        print(f"{k:<20}: {v}")


    #CLASSIFICATION

    print("CLASSIFICATION TRAINING")

    
    train_ds = BRISCClassificationDataset('data/processed', split='train', 
                                          transform=get_classification_transforms('train'))
    train_size = int(0.8 * len(train_ds))
    val_size = len(train_ds) - train_size
    train_sub, val_sub = random_split(train_ds, [train_size, val_size], 
                                      generator=torch.Generator().manual_seed(42))
    
    train_loader = DataLoader(train_sub, batch_size=HYPERPARAMS['BATCH_CLF'], 
                             shuffle=True, num_workers=4, pin_memory=True, 
                             persistent_workers=True)
    val_loader = DataLoader(val_sub, batch_size=HYPERPARAMS['BATCH_CLF'], 
                           shuffle=False, num_workers=2, pin_memory=True, 
                           persistent_workers=True)

    model_clf = UNetClassifier(1, 4).to(DEVICE)
    optimizer = optim.AdamW(model_clf.parameters(), lr=HYPERPARAMS['LR_CLF'], 
                          weight_decay=HYPERPARAMS['WEIGHT_DECAY'])
    criterion_clf = nn.CrossEntropyLoss(weight=class_weights)
    early_stopping = EarlyStopping(patience=HYPERPARAMS['EARLY_STOP_PATIENCE'], mode='max')
    
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_acc = 0

    for epoch in range(HYPERPARAMS['EPOCHS']):
        model_clf.train()
        epoch_loss = 0
        correct = 0
        total = 0
        
        for img, label in tqdm(train_loader, desc=f"Clf Epoch {epoch+1}/{HYPERPARAMS['EPOCHS']}"):
            img, label = img.to(DEVICE), label.to(DEVICE)
            optimizer.zero_grad()
            out = model_clf(img)
            loss = criterion_clf(out, label)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
            _, pred = torch.max(out, 1)
            correct += (pred == label).sum().item()
            total += label.size(0)
        
        train_acc = correct / total
        
        model_clf.eval()
        val_loss = 0
        all_preds, all_labels = [], []
        
        with torch.no_grad():
            for img, label in val_loader:
                img, label = img.to(DEVICE), label.to(DEVICE)
                out = model_clf(img)
                val_loss += criterion_clf(out, label).item()
                _, pred = torch.max(out, 1)
                all_preds.extend(pred.cpu().numpy())
                all_labels.extend(label.cpu().numpy())

        val_acc = (np.array(all_preds) == np.array(all_labels)).mean()
        
        history['train_loss'].append(epoch_loss / len(train_loader))
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss / len(val_loader))
        history['val_acc'].append(val_acc)
        
        print(f"  Epoch {epoch+1}: Train Acc={train_acc:.4f} | Val Acc={val_acc:.4f}")
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model_clf.state_dict(), 'models/unet_clf_best.pth')
            print(f" Saved best model (Val Acc: {val_acc:.4f})")
        
        if early_stopping(val_acc):
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

    plot_learning_curves(history, 'Classification', 'classification_results.png')
    

    print("\nEVALUATING CLASSIFICATION ON TEST SET...")
    test_ds = BRISCClassificationDataset('data/processed', split='test', 
                                         transform=get_classification_transforms('test'))
    test_loader = DataLoader(test_ds, batch_size=HYPERPARAMS['BATCH_CLF'], 
                            shuffle=False, num_workers=2, pin_memory=True)
    
    if os.path.exists('models/unet_clf_best.pth'):
        model_clf.load_state_dict(torch.load('models/unet_clf_best.pth'))
    
    results = evaluate_classification(model_clf, test_loader, criterion_clf, DEVICE)
    print(f"Test Acc: {results['acc']:.4f} | Prec: {results['precision']:.4f} | "
          f"Recall: {results['recall']:.4f} | F1: {results['f1']:.4f}")
    save_confusion_matrix(results['cm'], 'Classification Test CM', 'test_clf_cm.png')
    
    del train_loader, val_loader, test_loader, model_clf, optimizer
    torch.cuda.empty_cache()


    # SEGMENTATION

    print("SEGMENTATION TRAINING")
    
    train_ds = BRISCSegmentationDataset('data/processed', split='train', 
                                        transform=get_segmentation_transforms('train'))
    train_size = int(0.8 * len(train_ds))
    val_size = len(train_ds) - train_size
    train_sub, val_sub = random_split(train_ds, [train_size, val_size], 
                                      generator=torch.Generator().manual_seed(42))
    
    train_loader = DataLoader(train_sub, batch_size=HYPERPARAMS['BATCH_SEG'], 
                             shuffle=True, num_workers=4, pin_memory=True, 
                             persistent_workers=True)
    val_loader = DataLoader(val_sub, batch_size=HYPERPARAMS['BATCH_SEG'], 
                           shuffle=False, num_workers=2, pin_memory=True, 
                           persistent_workers=True)

    model_seg = UNet(1, 1).to(DEVICE)
    optimizer = optim.AdamW(model_seg.parameters(), lr=HYPERPARAMS['LR_SEG'], 
                          weight_decay=HYPERPARAMS['WEIGHT_DECAY'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5, factor=0.5)
    bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    dice_loss_fn = DiceLoss()
    early_stopping = EarlyStopping(patience=HYPERPARAMS['EARLY_STOP_PATIENCE'], mode='max')

    history = {'train_loss': [], 'val_loss': [], 'val_iou': [], 'val_dice': [], 'val_pixel_acc': []}
    best_iou = 0

    for epoch in range(HYPERPARAMS['EPOCHS']):
        model_seg.train()
        epoch_loss = 0
        
        for img, mask in tqdm(train_loader, desc=f"Seg Epoch {epoch+1}/{HYPERPARAMS['EPOCHS']}"):
            img, mask = img.to(DEVICE), mask.to(DEVICE)
            optimizer.zero_grad()
            out = model_seg(img)
            loss = bce(out, mask) + dice_loss_fn(out, mask)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        model_seg.eval()
        val_loss_sum = 0
        val_iou, val_dice, val_acc = 0, 0, 0
        
        with torch.no_grad():
            for img, mask in val_loader:
                img, mask = img.to(DEVICE), mask.to(DEVICE)
                out = model_seg(img)
                val_loss_sum += (bce(out, mask) + dice_loss_fn(out, mask)).item()
                pred = (torch.sigmoid(out) > 0.5).float()
                d, i, a = calculate_seg_metrics(pred, mask)
                val_iou += i
                val_dice += d
                val_acc += a

        avg_iou = val_iou / len(val_loader)
        avg_dice = val_dice / len(val_loader)
        avg_acc = val_acc / len(val_loader)

        history['train_loss'].append(epoch_loss / len(train_loader))
        history['val_loss'].append(val_loss_sum / len(val_loader))
        history['val_iou'].append(avg_iou)
        history['val_dice'].append(avg_dice)      
        history['val_pixel_acc'].append(avg_acc)
        
        print(f"  Epoch {epoch+1}: Val mIoU={avg_iou:.4f} | Dice={avg_dice:.4f} | "
              f"Pixel Acc={avg_acc:.4f}")
        
        scheduler.step(avg_iou)
        
        if avg_iou > best_iou:
            best_iou = avg_iou
            torch.save(model_seg.state_dict(), 'models/unet_seg_best.pth')
            print(f"Saved best model (Val mIoU: {avg_iou:.4f})")
        
        if early_stopping(avg_iou):
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

    plot_learning_curves(history, 'Segmentation', 'segmentation_metrics.png')
    
    
    print("\nEVALUATING SEGMENTATION ON TEST SET...")
    test_ds = BRISCSegmentationDataset('data/processed', split='test', 
                                       transform=get_segmentation_transforms('test'))
    test_loader = DataLoader(test_ds, batch_size=HYPERPARAMS['BATCH_SEG'], 
                            shuffle=False, num_workers=2, pin_memory=True)
    
    if os.path.exists('models/unet_seg_best.pth'):
        model_seg.load_state_dict(torch.load('models/unet_seg_best.pth'))
    
    results = evaluate_segmentation(model_seg, test_loader, bce, dice_loss_fn, DEVICE)
    print(f"Test mIoU: {results['iou']:.4f} | Dice: {results['dice']:.4f} | "
          f"Pixel Acc: {results['pixel_acc']:.4f}")
    

    del train_loader, val_loader, test_loader, model_seg, optimizer
    torch.cuda.empty_cache()


    #JOINT TRAINING

    print("JOINT TRAINING")

    
    full_joint_ds = NpyJointDataset('data/processed', split='train', 
                                    transform=get_segmentation_transforms('train'))
    train_size = int(0.8 * len(full_joint_ds))
    val_size = len(full_joint_ds) - train_size
    train_ds, val_ds = random_split(full_joint_ds, [train_size, val_size], 
                                    generator=torch.Generator().manual_seed(42))
    
    train_loader = DataLoader(train_ds, batch_size=HYPERPARAMS['BATCH_JOINT'], 
                             shuffle=True, num_workers=4, pin_memory=True, 
                             persistent_workers=True)
    val_loader = DataLoader(val_ds, batch_size=HYPERPARAMS['BATCH_JOINT'], 
                           shuffle=False, num_workers=2, pin_memory=True, 
                           persistent_workers=True)

    model_joint = UNetWithClassifier(1, 1, 4).to(DEVICE)
    optimizer = optim.AdamW(model_joint.parameters(), lr=HYPERPARAMS['LR_SEG'], 
                          weight_decay=HYPERPARAMS['WEIGHT_DECAY'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5, factor=0.5)
    bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    crit_clf = nn.CrossEntropyLoss(weight=class_weights)
    early_stopping = EarlyStopping(patience=HYPERPARAMS['EARLY_STOP_PATIENCE'], mode='max')
    
    history_joint = {
        'train_loss': [], 'train_acc': [], 'val_loss': [], 
        'val_acc': [], 'val_iou': [], 'val_dice': [], 'val_pixel_acc': []
    }
    best_score = 0

    for epoch in range(HYPERPARAMS['EPOCHS']):
        model_joint.train()
        epoch_loss = 0
        correct = 0
        total = 0
        
        for img, mask, label in tqdm(train_loader, desc=f"Joint Epoch {epoch+1}/{HYPERPARAMS['EPOCHS']}"):
            img, mask, label = img.to(DEVICE), mask.to(DEVICE), label.to(DEVICE)
            optimizer.zero_grad()
            s, c = model_joint(img)
            loss = (bce(s, mask) + dice_loss_fn(s, mask)) + crit_clf(c, label)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
            _, pc = torch.max(c, 1)
            correct += (pc == label).sum().item()
            total += label.size(0)

        train_acc = correct / total
        
        # Validation
        model_joint.eval()
        val_loss_sum = 0
        val_iou, val_dice, val_seg_acc, val_clf_acc = 0, 0, 0, 0
        
        with torch.no_grad():
            for img, mask, label in val_loader:
                img, mask, label = img.to(DEVICE), mask.to(DEVICE), label.to(DEVICE)
                s, c = model_joint(img)
                val_loss_sum += ((bce(s, mask) + dice_loss_fn(s, mask)) + crit_clf(c, label)).item()
                
                p = (torch.sigmoid(s) > 0.5).float()
                d, i, a = calculate_seg_metrics(p, mask)
                val_iou += i
                val_dice += d
                val_seg_acc += a
                
                _, pc = torch.max(c, 1)
                val_clf_acc += (pc == label).sum().item() / label.size(0)

        avg_clf_acc = val_clf_acc / len(val_loader)
        avg_iou = val_iou / len(val_loader)
        avg_dice = val_dice / len(val_loader)
        avg_seg_acc = val_seg_acc / len(val_loader)

        
        history_joint['train_loss'].append(epoch_loss / len(train_loader))
        history_joint['train_acc'].append(train_acc)
        history_joint['val_loss'].append(val_loss_sum / len(val_loader))
        history_joint['val_acc'].append(avg_clf_acc)
        history_joint['val_iou'].append(avg_iou)
        history_joint['val_dice'].append(avg_dice)
        history_joint['val_pixel_acc'].append(avg_seg_acc)
        
        print(f"  Epoch {epoch+1}: Clf Acc={avg_clf_acc:.4f} | mIoU={avg_iou:.4f} | "
              f"Dice={avg_dice:.4f} | PixAcc={avg_seg_acc:.4f}")
        
        score = (avg_iou + avg_clf_acc) / 2
        scheduler.step(score)
        
        if score > best_score:
            best_score = score
            torch.save(model_joint.state_dict(), 'models/unet_joint_best.pth')
            print(f"Saved best model (Combined Score: {score:.4f})")
        
        if early_stopping(score):
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

    plot_learning_curves(history_joint, 'Joint Training', 'joint_metrics.png')
    

    print("\nGenerating Joint Model Confusion Matrix...")
    if os.path.exists('models/unet_joint_best.pth'):
        model_joint.load_state_dict(torch.load('models/unet_joint_best.pth'))
    
    model_joint.eval()
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for img, mask, label in val_loader:
            img, label = img.to(DEVICE), label.to(DEVICE)
            _, c = model_joint(img)
            _, pred = torch.max(c, 1)
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(label.cpu().numpy())

    cm = confusion_matrix(all_labels, all_preds)
    save_confusion_matrix(cm, 'Joint Model Validation CM', 'joint_val_cm.png')
    
    # Test Evaluation
    print("\nEVALUATING JOINT MODEL ON TEST SET...")
    test_ds = NpyJointDataset('data/processed', split='test', 
                              transform=get_segmentation_transforms('test'))
    test_loader = DataLoader(test_ds, batch_size=HYPERPARAMS['BATCH_JOINT'], 
                            shuffle=False, num_workers=2, pin_memory=True)
    
    results = evaluate_joint(model_joint, test_loader, bce, dice_loss_fn, crit_clf, DEVICE)
    print(f"Test Clf Acc: {results['clf_acc']:.4f} | mIoU: {results['iou']:.4f} | "
          f"Dice: {results['dice']:.4f} | Pixel Acc: {results['pixel_acc']:.4f}")
    save_confusion_matrix(results['cm'], 'Joint Model Test CM', 'joint_test_cm.png')
    

    del train_loader, val_loader, test_loader, model_joint, optimizer
    torch.cuda.empty_cache()


    #ATTENTION U-NET

    print("ATTENTION U-NET TRAINING")

    
    # Reuse segmentation dataset
    train_ds = BRISCSegmentationDataset('data/processed', split='train', 
                                        transform=get_segmentation_transforms('train'))
    train_size = int(0.8 * len(train_ds))
    val_size = len(train_ds) - train_size
    train_sub, val_sub = random_split(train_ds, [train_size, val_size], 
                                      generator=torch.Generator().manual_seed(42))
    
    train_loader = DataLoader(train_sub, batch_size=HYPERPARAMS['BATCH_SEG'], 
                             shuffle=True, num_workers=4, pin_memory=True, 
                             persistent_workers=True)
    val_loader = DataLoader(val_sub, batch_size=HYPERPARAMS['BATCH_SEG'], 
                           shuffle=False, num_workers=2, pin_memory=True, 
                           persistent_workers=True)

    model_attn = AttentionUNet(1, 1).to(DEVICE)
    optimizer = optim.AdamW(model_attn.parameters(), lr=HYPERPARAMS['LR_SEG'], 
                          weight_decay=HYPERPARAMS['WEIGHT_DECAY'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5, factor=0.5)
    bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    dice_loss_fn = DiceLoss()
    early_stopping = EarlyStopping(patience=HYPERPARAMS['EARLY_STOP_PATIENCE'], mode='max')
    
    history = {'train_loss': [], 'val_loss': [], 'val_iou': [], 'val_dice': [], 'val_pixel_acc': []}
    best_iou = 0

    for epoch in range(HYPERPARAMS['EPOCHS']):
        # Training
        model_attn.train()
        epoch_loss = 0
        
        for img, mask in tqdm(train_loader, desc=f"Attn Epoch {epoch+1}/{HYPERPARAMS['EPOCHS']}"):
            img, mask = img.to(DEVICE), mask.to(DEVICE)
            optimizer.zero_grad()
            out = model_attn(img)
            loss = bce(out, mask) + dice_loss_fn(out, mask)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        # Validation
        model_attn.eval()
        val_loss_sum = 0
        val_iou, val_dice, val_acc = 0, 0, 0
        
        with torch.no_grad():
            for img, mask in val_loader:
                img, mask = img.to(DEVICE), mask.to(DEVICE)
                out = model_attn(img)
                val_loss_sum += (bce(out, mask) + dice_loss_fn(out, mask)).item()
                pred = (torch.sigmoid(out) > 0.5).float()
                d, i, a = calculate_seg_metrics(pred, mask)
                val_iou += i
                val_dice += d
                val_acc += a

        avg_iou = val_iou / len(val_loader)
        avg_dice = val_dice / len(val_loader)
        avg_acc = val_acc / len(val_loader)
        
        history['train_loss'].append(epoch_loss / len(train_loader))
        history['val_loss'].append(val_loss_sum / len(val_loader))
        history['val_iou'].append(avg_iou)
        history['val_dice'].append(avg_dice)
        history['val_pixel_acc'].append(avg_acc)
        
        print(f"Epoch {epoch+1}: Val mIoU={avg_iou:.4f} | Dice={avg_dice:.4f} | "
              f"Pixel Acc={avg_acc:.4f}")
        
        scheduler.step(avg_iou)
        
        if avg_iou > best_iou:
            best_iou = avg_iou
            torch.save(model_attn.state_dict(), 'models/att_unet_best.pth')
            print(f"Saved best model (Val mIoU: {avg_iou:.4f})")
        
        if early_stopping(avg_iou):
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

    plot_learning_curves(history, 'Attention U-Net', 'attention_metrics.png')
    

    print("\nEVALUATING ATTENTION U-NET ON TEST SET...")
    test_ds = BRISCSegmentationDataset('data/processed', split='test', 
                                       transform=get_segmentation_transforms('test'))
    test_loader = DataLoader(test_ds, batch_size=HYPERPARAMS['BATCH_SEG'], 
                            shuffle=False, num_workers=2, pin_memory=True)
    
    if os.path.exists('models/att_unet_best.pth'):
        model_attn.load_state_dict(torch.load('models/att_unet_best.pth'))
    
    results = evaluate_segmentation(model_attn, test_loader, bce, dice_loss_fn, DEVICE)
    print(f"Test mIoU: {results['iou']:.4f} | Dice: {results['dice']:.4f} | "
          f"Pixel Acc: {results['pixel_acc']:.4f}")
    

    del train_loader, val_loader, test_loader, model_attn, optimizer
    torch.cuda.empty_cache()
    