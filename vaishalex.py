#!/usr/bin/env python3

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

np.random.seed(42)
torch.manual_seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("using device:", device)

class config:
    base_path = Path('tomato_leaves/augmented_images')
    img_size = 224
    batch_size = 64
    epochs = 30
    lr = 0.001
    folds = 5
    weight_decay = 1e-4

cfg = config()

class tomato_dataset(Dataset):
    def __init__(self, paths, labels, transform=None):
        self.paths = paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert('RGB')
        label = self.labels[idx]
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(label)

class alexnet(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 11, 4, 2),
            nn.ReLU(True),
            nn.MaxPool2d(3, 2),
            nn.Conv2d(64, 192, 5, padding=2),
            nn.ReLU(True),
            nn.MaxPool2d(3, 2),
            nn.Conv2d(192, 384, 3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(384, 256, 3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(3, 2)
        )

        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))

        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

train_tf = transforms.Compose([
    transforms.Resize((cfg.img_size, cfg.img_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

val_tf = train_tf

def load_data():
    train_path = cfg.base_path / 'train'
    val_path = cfg.base_path / 'val'

    train_imgs, train_labels = [], []
    for p in sorted(train_path.glob('*.jpg')):
        train_imgs.append(str(p))
        train_labels.append(0 if p.stem.startswith('H') else 1)

    val_imgs, val_labels = [], []
    for p in sorted(val_path.glob('*.jpg')):
        val_imgs.append(str(p))
        val_labels.append(0 if p.stem.startswith('H') else 1)

    return (np.array(train_imgs), np.array(train_labels)), (np.array(val_imgs), np.array(val_labels))

def train_epoch(model, loader, loss_fn, opt, epoch):
    model.train()
    total_loss = 0
    preds_all, labels_all = [], []

    for imgs, labels in tqdm(loader, desc=f"epoch {epoch} train"):
        imgs, labels = imgs.to(device), labels.to(device)

        opt.zero_grad()
        out = model(imgs)
        loss = loss_fn(out, labels)
        loss.backward()
        opt.step()

        total_loss += loss.item()
        _, preds = torch.max(out, 1)
        preds_all.extend(preds.cpu().numpy())
        labels_all.extend(labels.cpu().numpy())

    acc = accuracy_score(labels_all, preds_all)
    return total_loss/len(loader), acc

def eval_epoch(model, loader, loss_fn, epoch):
    model.eval()
    total_loss = 0
    preds_all, labels_all = [], []

    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc=f"epoch {epoch} val"):
            imgs, labels = imgs.to(device), labels.to(device)
            out = model(imgs)
            loss = loss_fn(out, labels)

            total_loss += loss.item()
            _, preds = torch.max(out, 1)
            preds_all.extend(preds.cpu().numpy())
            labels_all.extend(labels.cpu().numpy())

    acc = accuracy_score(labels_all, preds_all)
    prec = precision_score(labels_all, preds_all, zero_division=0)
    rec = recall_score(labels_all, preds_all, zero_division=0)
    f1 = f1_score(labels_all, preds_all, zero_division=0)

    return total_loss/len(loader), acc, prec, rec, f1, labels_all, preds_all

def vaishnavi():

    (train_imgs, train_labels), (test_imgs, test_labels) = load_data()

    print("train:", len(train_imgs))
    print("test:", len(test_imgs))

    skf = StratifiedKFold(n_splits=cfg.folds, shuffle=True, random_state=42)
    fold_accs = []

    for fold, (tr_idx, val_idx) in enumerate(skf.split(train_imgs, train_labels), 1):

        print("\nfold", fold)

        tr_imgs, tr_labels = train_imgs[tr_idx], train_labels[tr_idx]
        val_imgs, val_labels = train_imgs[val_idx], train_labels[val_idx]

        tr_ds = tomato_dataset(tr_imgs, tr_labels, train_tf)
        val_ds = tomato_dataset(val_imgs, val_labels, val_tf)

        tr_loader = DataLoader(tr_ds, batch_size=cfg.batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=cfg.batch_size)

        model = alexnet().to(device)
        loss_fn = nn.CrossEntropyLoss()
        opt = optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

        best_acc = 0

        for epoch in range(1, cfg.epochs+1):
            tr_loss, tr_acc = train_epoch(model, tr_loader, loss_fn, opt, epoch)
            val_loss, val_acc, _, _, _, _, _ = eval_epoch(model, val_loader, loss_fn, epoch)

            print(f"epoch {epoch}: train_acc={tr_acc:.4f} val_acc={val_acc:.4f}")

            if val_acc > best_acc:
                best_acc = val_acc

        fold_accs.append(best_acc)
        print("best fold acc:", best_acc)

    print("\ncv mean:", np.mean(fold_accs))

    train_ds = tomato_dataset(train_imgs, train_labels, train_tf)
    test_ds = tomato_dataset(test_imgs, test_labels, val_tf)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size)

    model = alexnet().to(device)
    loss_fn = nn.CrossEntropyLoss()
    opt = optim.Adam(model.parameters(), lr=cfg.lr)

    train_accs, test_accs = [], []
    best_test = 0

    for epoch in range(1, cfg.epochs+1):
        tr_loss, tr_acc = train_epoch(model, train_loader, loss_fn, opt, epoch)
        te_loss, te_acc, te_prec, te_rec, te_f1, y_true, y_pred = eval_epoch(model, test_loader, loss_fn, epoch)

        train_accs.append(tr_acc)
        test_accs.append(te_acc)

        print(f"epoch {epoch}: train={tr_acc:.4f} test={te_acc:.4f}")

        if te_acc > best_test:
            best_test = te_acc
            torch.save(model.state_dict(), "best_model.pth")

    cm = confusion_matrix(y_true, y_pred)

    print("\nfinal accuracy:", best_test)
    print("precision:", te_prec)
    print("recall:", te_rec)
    print("f1:", te_f1)
    print("confusion matrix:\n", cm)

    plt.plot(train_accs, label="train")
    plt.plot(test_accs, label="test")
    plt.legend()
    plt.savefig("accuracy_curve.png")

    with open("results.json", "w") as f:
        json.dump({
            "cv_mean": float(np.mean(fold_accs)),
            "test_acc": float(best_test)
        }, f)

    print("done")

if __name__ == "__main__":
    vaishnavi()