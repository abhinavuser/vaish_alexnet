#!/usr/bin/env python3
"""
Complete Training Pipeline: Data Augmentation + AlexNet + Cross-Validation + Testing
For Tomato Leaves Disease Detection Dataset

Features:
- Image augmentation (rotation, brightness, zoom, etc.)
- AlexNet architecture implementation
- Stratified K-fold cross-validation
- Training with accuracy, precision, recall, F1-score
- Comprehensive evaluation and visualization
"""

import os
import json
import warnings
import numpy as np
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, confusion_matrix, classification_report, 
                             roc_auc_score, roc_curve, auc)
from PIL import Image
import tqdm

warnings.filterwarnings('ignore')

np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}\n")

class Config:
    base_path = Path('Tomato_Leaves')
    images_path = base_path / 'Images'
    labels_path = base_path / 'labels'
    
    img_size = 224  # AlexNet input size
    batch_size = 64  # Increased batch size for faster training
    num_epochs = 25  # Reduced from 50 for faster training
    learning_rate = 0.001
    num_folds = 3  # Reduced from 5 for faster cross-validation
    weight_decay = 1e-4
    
    # Class weights to handle imbalance (D=1.13x more likely)
    class_weights = torch.tensor([1.0, 1.13], dtype=torch.float32)

config = Config()

def get_augmentation_transforms():
    """Create augmentation transforms for training"""
    
    train_transform = transforms.Compose([
        transforms.RandomRotation(25),  # Rotate ±25 degrees
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),  # Color variations
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # Random shift
        transforms.RandomPerspective(distortion_scale=0.2, p=0.5),  # Perspective change
        transforms.RandomApply([
            transforms.GaussianBlur(kernel_size=3)
        ], p=0.3),  # Occasional blur
        transforms.RandomHorizontalFlip(p=0.5),  # Horizontal flip
        transforms.RandomVerticalFlip(p=0.3),  # Vertical flip
        transforms.Resize((config.img_size, config.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet normalization
                           std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((config.img_size, config.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform

class TomatoLeavesDataset(Dataset):
    """Custom Dataset for Tomato Leaves"""
    
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = torch.tensor(labels, dtype=torch.long)  # Ensure long type
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx].item()  # Get scalar value
        
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, label
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            # Return a dummy image on error
            return torch.zeros(3, config.img_size, config.img_size), label

class AlexNet(nn.Module):
    """AlexNet architecture for Tomato Leaves Classification"""
    
    def __init__(self, num_classes=2):
        super(AlexNet, self).__init__()
        
        self.features = nn.Sequential(
            # Conv1: 224x224x3 -> 55x55x64
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),  # -> 27x27x64
            
            # Conv2: 27x27x64 -> 27x27x192
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),  # -> 13x13x192
            
            # Conv3: 13x13x192 -> 13x13x384
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
            # Conv4: 13x13x384 -> 13x13x384
            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
            # Conv5: 13x13x384 -> 13x13x256
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),  # -> 6x6x256
        )
        
        # Average pooling
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        
        # Classification layers (Fully connected)
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

def load_dataset_info():
    """Load all images and their labels"""
    
    print("Loading dataset information...")
    
    image_paths = []
    labels = []
    
    # Load training images
    train_img_path = config.images_path / 'train'
    for img_file in sorted(train_img_path.glob('*.jpg')):
        image_paths.append(str(img_file))
        # Label: 0 for Healthy (H), 1 for Diseased (D)
        label = 0 if img_file.stem.startswith('H') else 1
        labels.append(label)
    
    # Load validation images
    val_img_path = config.images_path / 'val'
    val_images = []
    val_labels = []
    for img_file in sorted(val_img_path.glob('*.jpg')):
        val_images.append(str(img_file))
        label = 0 if img_file.stem.startswith('H') else 1
        val_labels.append(label)
    
    return (np.array(image_paths), np.array(labels)), (np.array(val_images), np.array(val_labels))

def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    progress_bar = tqdm.tqdm(train_loader, desc="Training")
    
    for images, labels in progress_bar:
        images = images.to(device)
        labels = torch.tensor(labels, dtype=torch.long).to(device)  # Ensure long type
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Track metrics
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        progress_bar.set_postfix({'loss': running_loss / len(all_preds)})
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = accuracy_score(all_labels, all_preds)
    
    return epoch_loss, epoch_acc

def validate(model, val_loader, criterion, device):
    """Validate the model"""
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = torch.tensor(labels, dtype=torch.long).to(device)  # Ensure long type
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            
            # Get predictions
            probs = torch.nn.functional.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs.data, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())  # Probability of class 1
    
    epoch_loss = running_loss / len(val_loader)
    epoch_acc = accuracy_score(all_labels, all_preds)
    
    # Calculate additional metrics
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    
    # ROC-AUC
    try:
        roc_auc = roc_auc_score(all_labels, all_probs)
    except:
        roc_auc = 0.0
    
    return (epoch_loss, epoch_acc, precision, recall, f1, roc_auc, 
            np.array(all_labels), np.array(all_preds), np.array(all_probs))

def main():
    """Main training pipeline with cross-validation"""
    
    # Load dataset
    (train_images, train_labels), (val_images, val_labels) = load_dataset_info()
    
    print(f"\n{'='*80}")
    print("TRAINING PIPELINE: TOMATO LEAVES DISEASE DETECTION WITH ALEXNET")
    print(f"{'='*80}\n")
    
    print(f"Training set: {len(train_images)} images")
    print(f"Validation set: {len(val_images)} images (reserved for final testing)")
    print(f"Total: {len(train_images) + len(val_images)} images\n")
    
    # Get transforms
    train_transform, val_transform = get_augmentation_transforms()
    
    # ============ CROSS-VALIDATION ON TRAINING SET ============
    print(f"{'='*80}")
    print(f"PHASE 1: STRATIFIED {config.num_folds}-FOLD CROSS-VALIDATION")
    print(f"{'='*80}\n")
    
    skf = StratifiedKFold(n_splits=config.num_folds, shuffle=True, random_state=42)
    
    cv_results = {
        'fold_accuracies': [],
        'fold_val_accuracies': [],
        'fold_precisions': [],
        'fold_recalls': [],
        'fold_f1_scores': [],
        'fold_roc_aucs': [],
        'final_model': None
    }
    
    best_cv_acc = 0.0
    best_cv_model_state = None
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(train_images, train_labels)):
        print(f"\nFold {fold + 1}/{config.num_folds}")
        print("-" * 80)
        
        # Split data
        fold_train_images = train_images[train_idx]
        fold_train_labels = train_labels[train_idx]
        fold_val_images = train_images[val_idx]
        fold_val_labels = train_labels[val_idx]
        
        # Create datasets
        train_dataset = TomatoLeavesDataset(fold_train_images, fold_train_labels, train_transform)
        val_dataset = TomatoLeavesDataset(fold_val_images, fold_val_labels, val_transform)
        
        # Create dataloaders
        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, 
                                 shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=config.batch_size, 
                               shuffle=False, num_workers=0)
        
        # Initialize model
        model = AlexNet(num_classes=2).to(DEVICE)
        
        # Loss and optimizer with class weights
        class_weights = config.class_weights.to(DEVICE)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, 
                              weight_decay=config.weight_decay)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
        
        # Training loop
        best_val_acc = 0.0
        patience = 5  # Reduced for faster stopping
        patience_counter = 0
        
        train_losses = []
        train_accs = []
        val_losses = []
        val_accs = []
        
        for epoch in range(config.num_epochs):
            train_loss, train_acc = train_epoch(model, train_loader, criterion, 
                                               optimizer, DEVICE)
            
            val_result = validate(model, val_loader, criterion, DEVICE)
            val_loss, val_acc, val_precision, val_recall, val_f1, val_roc_auc, _, _, _ = val_result
            
            train_losses.append(train_loss)
            train_accs.append(train_acc)
            val_losses.append(val_loss)
            val_accs.append(val_acc)
            
            scheduler.step()
            
            if (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch+1:3d}/{config.num_epochs} | "
                      f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                      f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
            
            # Early stopping
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"\n  Early stopping at epoch {epoch+1}")
                    break
        
        # Evaluate on validation fold
        val_result = validate(model, val_loader, criterion, DEVICE)
        _, val_acc, val_precision, val_recall, val_f1, val_roc_auc, _, _, _ = val_result
        
        cv_results['fold_accuracies'].append(best_val_acc)
        cv_results['fold_val_accuracies'].append(val_acc)
        cv_results['fold_precisions'].append(val_precision)
        cv_results['fold_recalls'].append(val_recall)
        cv_results['fold_f1_scores'].append(val_f1)
        cv_results['fold_roc_aucs'].append(val_roc_auc)
        
        print(f"\n  Fold Results:")
        print(f"    Accuracy:  {val_acc:.4f}")
        print(f"    Precision: {val_precision:.4f}")
        print(f"    Recall:    {val_recall:.4f}")
        print(f"    F1-Score:  {val_f1:.4f}")
        print(f"    ROC-AUC:   {val_roc_auc:.4f}")
        
        # Save best model from CV
        if val_acc > best_cv_acc:
            best_cv_acc = val_acc
            best_cv_model_state = model.state_dict()
    
    # ============ CROSS-VALIDATION SUMMARY ============
    print(f"\n{'='*80}")
    print("PHASE 1 RESULTS: CROSS-VALIDATION SUMMARY")
    print(f"{'='*80}\n")
    
    cv_acc_mean = np.mean(cv_results['fold_accuracies'])
    cv_acc_std = np.std(cv_results['fold_accuracies'])
    cv_val_acc_mean = np.mean(cv_results['fold_val_accuracies'])
    cv_val_acc_std = np.std(cv_results['fold_val_accuracies'])
    
    print(f"Cross-Validation Accuracy (Overall): {cv_acc_mean:.4f} ± {cv_acc_std:.4f}")
    print(f"Cross-Validation Accuracy (Final):   {cv_val_acc_mean:.4f} ± {cv_val_acc_std:.4f}")
    print(f"Precision (Mean):                    {np.mean(cv_results['fold_precisions']):.4f} ± {np.std(cv_results['fold_precisions']):.4f}")
    print(f"Recall (Mean):                       {np.mean(cv_results['fold_recalls']):.4f} ± {np.std(cv_results['fold_recalls']):.4f}")
    print(f"F1-Score (Mean):                     {np.mean(cv_results['fold_f1_scores']):.4f} ± {np.std(cv_results['fold_f1_scores']):.4f}")
    print(f"ROC-AUC (Mean):                      {np.mean(cv_results['fold_roc_aucs']):.4f} ± {np.std(cv_results['fold_roc_aucs']):.4f}")
    
    if cv_val_acc_mean >= 0.90:
        print(f"\n✓ CROSS-VALIDATION ACCURACY TARGET ACHIEVED: {cv_val_acc_mean:.2%} (≥ 90%)")
    else:
        print(f"\n⚠ Cross-validation accuracy: {cv_val_acc_mean:.2%} (Target: ≥ 90%)")
    
    # ============ FINAL TRAINING ON COMPLETE TRAINING SET ============
    print(f"\n{'='*80}")
    print("PHASE 2: FINAL TRAINING ON COMPLETE TRAINING SET")
    print(f"{'='*80}\n")
    
    # Create final dataset with all training data
    final_train_dataset = TomatoLeavesDataset(train_images, train_labels, train_transform)
    final_train_loader = DataLoader(final_train_dataset, batch_size=config.batch_size, 
                                   shuffle=True, num_workers=0)
    
    # Create validation dataset for holdout testing
    val_dataset = TomatoLeavesDataset(val_images, val_labels, val_transform)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, 
                           shuffle=False, num_workers=0)
    
    # Initialize final model or load best CV model
    final_model = AlexNet(num_classes=2).to(DEVICE)
    if best_cv_model_state is not None:
        final_model.load_state_dict(best_cv_model_state)
    
    class_weights = config.class_weights.to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(final_model.parameters(), lr=config.learning_rate, 
                          weight_decay=config.weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    
    best_val_acc_final = 0.0
    patience = 7  # Reduced for faster training
    patience_counter = 0
    
    final_train_losses = []
    final_train_accs = []
    final_val_losses = []
    final_val_accs = []
    
    for epoch in range(config.num_epochs):
        train_loss, train_acc = train_epoch(final_model, final_train_loader, criterion, 
                                           optimizer, DEVICE)
        
        val_result = validate(final_model, val_loader, criterion, DEVICE)
        val_loss, val_acc, _, _, _, _, _, _, _ = val_result
        
        final_train_losses.append(train_loss)
        final_train_accs.append(train_acc)
        final_val_losses.append(val_loss)
        final_val_accs.append(val_acc)
        
        scheduler.step()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:3d}/{config.num_epochs} | "
                  f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
        
        if val_acc > best_val_acc_final:
            best_val_acc_final = val_acc
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break
    
    cv_results['final_model'] = final_model
    
    # ============ FINAL EVALUATION ON TEST SET (Validation Set) ============
    print(f"\n{'='*80}")
    print("PHASE 3: FINAL EVALUATION ON TEST SET (VALIDATION IMAGES)")
    print(f"{'='*80}\n")
    
    final_model.eval()
    test_preds = []
    test_labels = []
    test_probs = []
    
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            
            outputs = final_model(images)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs.data, 1)
            
            test_preds.extend(predicted.cpu().numpy())
            test_labels.extend(labels.cpu().numpy())
            test_probs.extend(probs[:, 1].cpu().numpy())
    
    test_preds = np.array(test_preds)
    test_labels = np.array(test_labels)
    test_probs = np.array(test_probs)
    
    # Calculate metrics
    test_acc = accuracy_score(test_labels, test_preds)
    test_precision = precision_score(test_labels, test_preds, zero_division=0)
    test_recall = recall_score(test_labels, test_preds, zero_division=0)
    test_f1 = f1_score(test_labels, test_preds, zero_division=0)
    test_roc_auc = roc_auc_score(test_labels, test_probs)
    
    print(f"\nTest Set Metrics:")
    print(f"  Accuracy:   {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"  Precision:  {test_precision:.4f}")
    print(f"  Recall:     {test_recall:.4f}")
    print(f"  F1-Score:   {test_f1:.4f}")
    print(f"  ROC-AUC:    {test_roc_auc:.4f}")
    
    if test_acc >= 0.90:
        print(f"\n✓ TEST ACCURACY TARGET ACHIEVED: {test_acc:.2%} (≥ 90%)")
    else:
        print(f"\n⚠ Test accuracy: {test_acc:.2%} (Target: ≥ 90%)")
    
    # ============ DETAILED CLASSIFICATION REPORT ============
    print(f"\n{'='*80}")
    print("DETAILED CLASSIFICATION REPORT")
    print(f"{'='*80}\n")
    
    class_names = ['Healthy', 'Diseased']
    print(classification_report(test_labels, test_preds, target_names=class_names))
    
    # Confusion Matrix
    cm = confusion_matrix(test_labels, test_preds)
    print(f"\nConfusion Matrix:")
    print(f"                Predicted Healthy  Predicted Diseased")
    print(f"Actual Healthy        {cm[0, 0]:4d}           {cm[0, 1]:4d}")
    print(f"Actual Diseased       {cm[1, 0]:4d}           {cm[1, 1]:4d}")
    
    # ============ SAVE RESULTS ============
    results = {
        'cross_validation': {
            'num_folds': config.num_folds,
            'fold_accuracies': cv_results['fold_accuracies'],
            'fold_val_accuracies': cv_results['fold_val_accuracies'],
            'fold_precisions': cv_results['fold_precisions'],
            'fold_recalls': cv_results['fold_recalls'],
            'fold_f1_scores': cv_results['fold_f1_scores'],
            'fold_roc_aucs': cv_results['fold_roc_aucs'],
            'mean_accuracy': cv_val_acc_mean,
            'std_accuracy': cv_val_acc_std,
            'mean_precision': float(np.mean(cv_results['fold_precisions'])),
            'mean_recall': float(np.mean(cv_results['fold_recalls'])),
            'mean_f1': float(np.mean(cv_results['fold_f1_scores'])),
            'mean_roc_auc': float(np.mean(cv_results['fold_roc_aucs'])),
        },
        'test_results': {
            'test_accuracy': float(test_acc),
            'test_precision': float(test_precision),
            'test_recall': float(test_recall),
            'test_f1': float(test_f1),
            'test_roc_auc': float(test_roc_auc),
            'confusion_matrix': cm.tolist(),
            'num_test_samples': len(test_labels),
            'num_healthy_test': int(np.sum(test_labels == 0)),
            'num_diseased_test': int(np.sum(test_labels == 1)),
        },
        'training_config': {
            'image_size': config.img_size,
            'batch_size': config.batch_size,
            'num_epochs': config.num_epochs,
            'learning_rate': config.learning_rate,
            'weight_decay': config.weight_decay,
            'model': 'AlexNet',
            'device': str(DEVICE),
        },
        'data_augmentation': [
            'Random Rotation (±25°)',
            'Color Jitter (brightness, contrast, saturation)',
            'Random Affine (translation)',
            'Random Perspective',
            'Gaussian Blur (occasional)',
            'Horizontal Flip (50%)',
            'Vertical Flip (30%)',
        ]
    }
    
    # Save results to JSON
    with open('training_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*80}")
    print("✓ Training Results saved to 'training_results.json'")
    print(f"{'='*80}\n")
    
    # ============ SAVE MODEL ============
    model_path = 'alexnet_trained_model.pth'
    torch.save(final_model.state_dict(), model_path)
    print(f"✓ Model saved to '{model_path}'")
    
    return final_model, results, (test_labels, test_preds, test_probs)

if __name__ == '__main__':
    model, results, test_data = main()
    
    # Generate visualization
    print("\nGenerating visualizations...")
    from visualize_results import visualize_all
    visualize_all(results, test_data)
    print("✓ Visualizations saved")
