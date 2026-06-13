"""
Kisan AI - Model Training Script
EfficientNetB0 fine-tuned on PlantVillage dataset (38 classes)

Usage:
    python model/train.py --data_dir data/plantvillage/color --epochs 15 --batch_size 32
"""

import os
import json
import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns


# ─── Config ───────────────────────────────────────────────────────────────────

DEFAULT_DATA_DIR  = "data/plantvillage/color"
DEFAULT_MODEL_DIR = "model"
IMG_SIZE          = 224
NUM_CLASSES       = 38
SEED              = 42

# ImageNet normalization (required for EfficientNet pretrained weights)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


# ─── Transforms ───────────────────────────────────────────────────────────────

def get_transforms():
    train_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(p=0.1),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return train_tf, val_tf


# ─── Dataset ──────────────────────────────────────────────────────────────────

def load_datasets(data_dir, train_tf, val_tf):
    full = datasets.ImageFolder(data_dir, transform=train_tf)
    n    = len(full)
    train_n = int(0.8 * n)
    val_n   = int(0.1 * n)
    test_n  = n - train_n - val_n

    generator = torch.Generator().manual_seed(SEED)
    train_set, val_set, test_set = random_split(
        full, [train_n, val_n, test_n], generator=generator
    )

    # Apply val transforms to val and test
    val_set.dataset  = datasets.ImageFolder(data_dir, transform=val_tf)
    test_set.dataset = datasets.ImageFolder(data_dir, transform=val_tf)

    print(f"\nDataset split:")
    print(f"  Train : {len(train_set):,} images")
    print(f"  Val   : {len(val_set):,} images")
    print(f"  Test  : {len(test_set):,} images")
    print(f"  Classes: {len(full.classes)}")

    return train_set, val_set, test_set, full.classes


# ─── Model ────────────────────────────────────────────────────────────────────

def build_model(num_classes=NUM_CLASSES, freeze_backbone=True):
    model = models.efficientnet_b0(weights='IMAGENET1K_V1')

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    # Replace the classifier head
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, num_classes)
    )
    return model


# ─── Training ─────────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for batch_idx, (imgs, labels) in enumerate(loader):
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        correct    += (outputs.argmax(1) == labels).sum().item()
        total      += labels.size(0)

        if (batch_idx + 1) % 50 == 0:
            print(f"  Batch {batch_idx+1}/{len(loader)}  "
                  f"loss={total_loss/(batch_idx+1):.4f}  "
                  f"acc={correct/total:.4f}")

    return total_loss / len(loader), correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = model(imgs)
        loss = criterion(outputs, labels)

        total_loss += loss.item()
        correct    += (outputs.argmax(1) == labels).sum().item()
        total      += labels.size(0)

    return total_loss / len(loader), correct / total


# ─── Evaluation plots ─────────────────────────────────────────────────────────

def save_confusion_matrix(model, test_loader, class_names, device, save_path):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(device)
            preds = model(imgs).argmax(1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    cm = confusion_matrix(all_labels, all_preds)
    short_names = [c.split("___")[1].replace("_", " ")[:15] for c in class_names]

    plt.figure(figsize=(18, 16))
    sns.heatmap(cm, annot=False, cmap='Greens',
                xticklabels=short_names, yticklabels=short_names)
    plt.title("Kisan AI — Confusion Matrix (Test Set)", fontsize=14)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.xticks(rotation=45, ha='right', fontsize=7)
    plt.yticks(rotation=0, fontsize=7)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"\nConfusion matrix saved → {save_path}")

    report = classification_report(all_labels, all_preds,
                                   target_names=short_names)
    print("\nClassification Report:")
    print(report)

    report_path = Path(save_path).parent / "classification_report.txt"
    with open(report_path, "w") as f:
        f.write(report)
    print(f"Classification report saved → {report_path}")


def save_training_curves(history, save_path):
    epochs = range(1, len(history['train_acc']) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(epochs, history['train_loss'], 'b-o', label='Train loss', markersize=4)
    ax1.plot(epochs, history['val_loss'],   'r-o', label='Val loss',   markersize=4)
    ax1.set_title('Loss')
    ax1.set_xlabel('Epoch')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, history['train_acc'], 'b-o', label='Train acc', markersize=4)
    ax2.plot(epochs, history['val_acc'],   'r-o', label='Val acc',   markersize=4)
    ax2.set_title('Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylim([0, 1])
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.suptitle('Kisan AI — Training Curves', fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Training curves saved → {save_path}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("No GPU detected — training will be slow.")
        print("Recommendation: Use Google Colab (Runtime > Change runtime type > GPU)")

    # Dataset
    train_tf, val_tf = get_transforms()
    train_set, val_set, test_set, class_names = load_datasets(
        args.data_dir, train_tf, val_tf
    )

    train_loader = DataLoader(train_set, batch_size=args.batch_size,
                              shuffle=True,  num_workers=args.num_workers, pin_memory=True)
    val_loader   = DataLoader(val_set,   batch_size=args.batch_size,
                              shuffle=False, num_workers=args.num_workers, pin_memory=True)
    test_loader  = DataLoader(test_set,  batch_size=args.batch_size,
                              shuffle=False, num_workers=args.num_workers, pin_memory=True)

    # Save class index mapping
    os.makedirs(args.model_dir, exist_ok=True)
    class_map_path = Path(args.model_dir) / "class_names.json"
    with open(class_map_path, "w") as f:
        json.dump(class_names, f, indent=2)
    print(f"Class names saved → {class_map_path}")

    # Model
    model = build_model(num_classes=len(class_names), freeze_backbone=True).to(device)
    criterion = nn.CrossEntropyLoss()

    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    best_val_acc = 0.0
    model_save_path = Path(args.model_dir) / "best_model.pth"

    # ── Phase 1: Train head only (5 epochs) ──────────────────────────────────
    print("\n" + "="*60)
    print("PHASE 1 — Training classifier head only (5 epochs)")
    print("="*60)

    optimizer  = optim.Adam(model.classifier.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler  = CosineAnnealingLR(optimizer, T_max=5)

    for epoch in range(5):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss,   val_acc   = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        elapsed = time.time() - t0
        print(f"\nEpoch {epoch+1}/5 ({elapsed:.0f}s)  "
              f"train_loss={train_loss:.4f}  train_acc={train_acc:.4f}  "
              f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), model_save_path)
            print(f"  ✓ New best model saved (val_acc={val_acc:.4f})")

    # ── Phase 2: Unfreeze all layers ─────────────────────────────────────────
    print("\n" + "="*60)
    print(f"PHASE 2 — Full fine-tuning ({args.epochs - 5} epochs)")
    print("="*60)

    for param in model.parameters():
        param.requires_grad = True

    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs - 5)

    for epoch in range(args.epochs - 5):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss,   val_acc   = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        elapsed = time.time() - t0
        print(f"\nEpoch {epoch+6}/{args.epochs} ({elapsed:.0f}s)  "
              f"train_loss={train_loss:.4f}  train_acc={train_acc:.4f}  "
              f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), model_save_path)
            print(f"  ✓ New best model saved (val_acc={val_acc:.4f})")

    print(f"\nTraining complete. Best val accuracy: {best_val_acc:.4f}")

    # ── Evaluation ────────────────────────────────────────────────────────────
    print("\nRunning final evaluation on test set...")
    model.load_state_dict(torch.load(model_save_path, map_location=device))

    save_confusion_matrix(
        model, test_loader, class_names, device,
        save_path=str(Path(args.model_dir) / "confusion_matrix.png")
    )
    save_training_curves(
        history,
        save_path=str(Path(args.model_dir) / "training_curves.png")
    )

    print(f"\nAll done! Model saved at: {model_save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Kisan AI crop disease model")
    parser.add_argument("--data_dir",    default=DEFAULT_DATA_DIR)
    parser.add_argument("--model_dir",   default=DEFAULT_MODEL_DIR)
    parser.add_argument("--epochs",      type=int, default=15)
    parser.add_argument("--batch_size",  type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=2)
    args = parser.parse_args()
    main(args)
