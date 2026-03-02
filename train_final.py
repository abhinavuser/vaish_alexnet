#!/usr/bin/env python3
"""
AlexNet Training with Clear Epoch Display, Cross-Validation, and Metrics
Training on augmented balanced dataset
"""

import os
import json
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from PIL import Image
from tqdm import tqdm

# Set seeds
np.random.seed(42)
torch.manual_seed(42)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}\n")

# ==================== CONFIG ====================
class Config:
    base_path = Path('Tomato_Leaves/Augmented_Images')
    img_size = 224
    batch_size = 64
    num_epochs = 30
    learning_rate = 0.001
    num_folds = 5
    weight_decay = 1e-4

config = Config()

# ==================== DATASET ====================
class TomatoDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            img = self.transform(img)
        
        return img, torch.tensor(label, dtype=torch.long)

# ==================== ALEXNET ====================
class AlexNet(nn.Module):
    def __init__(self, num_classes=2):
        super(AlexNet, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# ==================== TRANSFORMS ====================
train_transform = transforms.Compose([
    transforms.Resize((config.img_size, config.img_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((config.img_size, config.img_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ==================== LOAD DATA ====================
def load_data():
    train_path = config.base_path / 'train'
    val_path = config.base_path / 'val'
    
    # Load training images
    train_images = []
    train_labels = []
    for img_path in sorted(train_path.glob('*.jpg')):
        train_images.append(str(img_path))
        label = 0 if img_path.stem.startswith('H') else 1
        train_labels.append(label)
    
    # Load validation images
    val_images = []
    val_labels = []
    for img_path in sorted(val_path.glob('*.jpg')):
        val_images.append(str(img_path))
        label = 0 if img_path.stem.startswith('H') else 1
        val_labels.append(label)
    
    return (np.array(train_images), np.array(train_labels)), (np.array(val_images), np.array(val_labels))

# ==================== TRAINING ====================
def train_one_epoch(model, loader, criterion, optimizer, device, epoch):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    pbar = tqdm(loader, desc=f"Epoch {epoch:2d} [TRAIN]")
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    epoch_loss = running_loss / len(loader)
    epoch_acc = accuracy_score(all_labels, all_preds)
    
    return epoch_loss, epoch_acc

def validate(model, loader, criterion, device, epoch):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        pbar = tqdm(loader, desc=f"Epoch {epoch:2d} [VAL]  ")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    epoch_loss = running_loss / len(loader)
    epoch_acc = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    
    return epoch_loss, epoch_acc, precision, recall, f1, np.array(all_labels), np.array(all_preds)

# ==================== MAIN ====================
def main():
    print("="*80)
    print("ALEXNET TRAINING ON AUGMENTED TOMATO LEAVES DATASET")
    print("="*80)
    
    # Load data
    (train_images, train_labels), (test_images, test_labels) = load_data()
    
    healthy_count = np.sum(train_labels == 0)
    diseased_count = np.sum(train_labels == 1)
    
    print(f"\nDataset Loaded:")
    print(f"  Training:   {len(train_images)} images")
    print(f"    - Healthy:  {healthy_count}")
    print(f"    - Diseased: {diseased_count}")
    print(f"    - Balance:  {healthy_count/diseased_count:.2f}")
    print(f"  Test Set:   {len(test_images)} images")
    
    # ==================== CROSS-VALIDATION ====================
    print(f"\n{'='*80}")
    print(f"PHASE 1: {config.num_folds}-FOLD CROSS-VALIDATION")
    print(f"{'='*80}\n")
    
    skf = StratifiedKFold(n_splits=config.num_folds, shuffle=True, random_state=42)
    cv_results = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(train_images, train_labels), 1):
        print(f"\n{'─'*80}")
        print(f"FOLD {fold}/{config.num_folds}")
        print(f"{'─'*80}")
        
        # Split data
        fold_train_imgs = train_images[train_idx]
        fold_train_labels = train_labels[train_idx]
        fold_val_imgs = train_images[val_idx]
        fold_val_labels = train_labels[val_idx]
        
        print(f"Train: {len(fold_train_imgs)} | Val: {len(fold_val_imgs)}")
        
        # Create datasets
        train_dataset = TomatoDataset(fold_train_imgs, fold_train_labels, train_transform)
        val_dataset = TomatoDataset(fold_val_imgs, fold_val_labels, val_transform)
        
        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=0)
        
        # Model
        model = AlexNet(num_classes=2).to(DEVICE)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
        
        # Training
        best_val_acc = 0.0
        train_accs = []
        val_accs = []
        
        for epoch in range(1, config.num_epochs + 1):
            train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE, epoch)
            val_loss, val_acc, val_prec, val_recall, val_f1, _, _ = validate(model, val_loader, criterion, DEVICE, epoch)
            
            train_accs.append(train_acc)
            val_accs.append(val_acc)
            
            # Display epoch results
            print(f"  Epoch {epoch:2d}: Train Acc = {train_acc:.4f} | Val Acc = {val_acc:.4f} | "
                  f"Train Loss = {train_loss:.4f} | Val Loss = {val_loss:.4f}")
            
            scheduler.step()
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
        
        print(f"\n  Fold {fold} Best Val Accuracy: {best_val_acc:.4f} ({best_val_acc*100:.2f}%)")
        
        cv_results.append({
            'fold': fold,
            'best_val_acc': best_val_acc,
            'train_accs': train_accs,
            'val_accs': val_accs
        })
    
    # ==================== CV SUMMARY ====================
    print(f"\n{'='*80}")
    print("CROSS-VALIDATION SUMMARY")
    print(f"{'='*80}\n")
    
    cv_accs = [r['best_val_acc'] for r in cv_results]
    mean_cv_acc = np.mean(cv_accs)
    std_cv_acc = np.std(cv_accs)
    
    print("Fold Results:")
    for i, acc in enumerate(cv_accs, 1):
        print(f"  Fold {i}: {acc:.4f} ({acc*100:.2f}%)")
    
    print(f"\nCross-Validation Accuracy: {mean_cv_acc:.4f} ± {std_cv_acc:.4f}")
    print(f"                           ({mean_cv_acc*100:.2f}% ± {std_cv_acc*100:.2f}%)")
    
    if mean_cv_acc >= 0.90:
        print(f"\n✓ TARGET ACHIEVED: {mean_cv_acc*100:.2f}% >= 90%")
    else:
        print(f"\n⚠ Target not met: {mean_cv_acc*100:.2f}% < 90%")
    
    # ==================== FINAL TRAINING ON FULL DATASET ====================
    print(f"\n{'='*80}")
    print("PHASE 2: FINAL TRAINING ON COMPLETE DATASET")
    print(f"{'='*80}\n")
    
    train_dataset = TomatoDataset(train_images, train_labels, train_transform)
    test_dataset = TomatoDataset(test_images, test_labels, val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=0)
    
    final_model = AlexNet(num_classes=2).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(final_model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    
    best_test_acc = 0.0
    final_train_accs = []
    final_test_accs = []
    
    for epoch in range(1, config.num_epochs + 1):
        train_loss, train_acc = train_one_epoch(final_model, train_loader, criterion, optimizer, DEVICE, epoch)
        test_loss, test_acc, test_prec, test_recall, test_f1, _, _ = validate(final_model, test_loader, criterion, DEVICE, epoch)
        
        final_train_accs.append(train_acc)
        final_test_accs.append(test_acc)
        
        print(f"  Epoch {epoch:2d}: Train Acc = {train_acc:.4f} | Test Acc = {test_acc:.4f} | "
              f"Train Loss = {train_loss:.4f} | Test Loss = {test_loss:.4f}")
        
        scheduler.step()
        
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            torch.save(final_model.state_dict(), 'best_alexnet_model.pth')
    
    # ==================== FINAL TEST ====================
    print(f"\n{'='*80}")
    print("PHASE 3: FINAL TEST SET EVALUATION")
    print(f"{'='*80}\n")
    
    final_model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = final_model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    test_acc = accuracy_score(all_labels, all_preds)
    test_prec = precision_score(all_labels, all_preds, zero_division=0)
    test_recall = recall_score(all_labels, all_preds, zero_division=0)
    test_f1 = f1_score(all_labels, all_preds, zero_division=0)
    cm = confusion_matrix(all_labels, all_preds)
    
    print(f"Final Test Metrics:")
    print(f"  Accuracy:  {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"  Precision: {test_prec:.4f}")
    print(f"  Recall:    {test_recall:.4f}")
    print(f"  F1-Score:  {test_f1:.4f}")
    
    print(f"\nConfusion Matrix:")
    print(f"                Predicted")
    print(f"              H      D")
    print(f"Actual H    {cm[0,0]:3d}   {cm[0,1]:3d}")
    print(f"       D    {cm[1,0]:3d}   {cm[1,1]:3d}")
    
    if test_acc >= 0.90:
        print(f"\n✓ FINAL TEST TARGET ACHIEVED: {test_acc*100:.2f}% >= 90%")
    else:
        print(f"\n⚠ Test accuracy: {test_acc*100:.2f}% < 90%")
    
    # ==================== SAVE RESULTS ====================
    results = {
        'cross_validation': {
            'num_folds': config.num_folds,
            'fold_accuracies': cv_accs,
            'mean_accuracy': float(mean_cv_acc),
            'std_accuracy': float(std_cv_acc),
        },
        'final_test': {
            'accuracy': float(test_acc),
            'precision': float(test_prec),
            'recall': float(test_recall),
            'f1_score': float(test_f1),
            'confusion_matrix': cm.tolist(),
        },
        'training_history': {
            'train_accuracies': final_train_accs,
            'test_accuracies': final_test_accs,
        }
    }
    
    with open('final_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*80}")
    print("✓ Training Complete!")
    print(f"✓ Results saved to: final_results.json")
    print(f"✓ Model saved to: best_alexnet_model.pth")
    print(f"{'='*80}\n")
    
    # Plot accuracy curves
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    epochs = range(1, len(final_train_accs) + 1)
    plt.plot(epochs, final_train_accs, 'b-', label='Train Accuracy', linewidth=2)
    plt.plot(epochs, final_test_accs, 'r-', label='Test Accuracy', linewidth=2)
    plt.axhline(y=0.90, color='g', linestyle='--', label='90% Target')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training vs Test Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.bar(['CV Accuracy', 'Test Accuracy'], [mean_cv_acc, test_acc], color=['blue', 'green'])
    plt.axhline(y=0.90, color='r', linestyle='--', label='90% Target')
    plt.ylabel('Accuracy')
    plt.title('Final Metrics')
    plt.ylim([0, 1])
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('training_accuracy_curves.png', dpi=300)
    print("✓ Accuracy curves saved to: training_accuracy_curves.png\n")

if __name__ == '__main__':
    main()
