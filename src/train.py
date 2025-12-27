# src/train.py

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import torchvision.transforms as transforms
from src.dataset import CORROSIONDataset
from src.model import CorrosionCNN
from sklearn.metrics import accuracy_score

# -------------------------------
# Configuration
# -------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 16
NUM_EPOCHS = 10
LEARNING_RATE = 1e-4
DATA_DIR = "data/processed"
CHECKPOINT_DIR = "models/"
QUICK_TEST = False  # Set True for small subset testing

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# -------------------------------
# Transforms
# -------------------------------
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# -------------------------------
# MAIN
# -------------------------------
if __name__ == "__main__":

    # -------------------------------
    # Datasets
    # -------------------------------
    train_dataset = CORROSIONDataset(os.path.join(DATA_DIR, "train"), transform=train_transforms)
    val_dataset = CORROSIONDataset(os.path.join(DATA_DIR, "val"), transform=val_transforms)

    # Quick test: take small subset
    if QUICK_TEST:
        train_dataset = Subset(train_dataset, range(min(20, len(train_dataset))))
        val_dataset = Subset(val_dataset, range(min(20, len(val_dataset))))
        BATCH_SIZE = 4
        NUM_EPOCHS = 2

    # -------------------------------
    # DataLoaders
    # -------------------------------
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # -------------------------------
    # Model, Loss, Optimizer
    # -------------------------------
    model = CorrosionCNN(pretrained=True).to(DEVICE)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    best_val_acc = 0

    # -------------------------------
    # Training Loop
    # -------------------------------
    for epoch in range(NUM_EPOCHS):
        model.train()
        train_losses = []
        train_preds = []
        train_labels = []

        for images, labels in train_loader:
            images = images.to(DEVICE)
            labels = labels.float().unsqueeze(1).to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())
            train_preds.extend((outputs.detach().cpu() > 0.5).int().numpy())
            train_labels.extend(labels.detach().cpu().numpy())

        train_acc = accuracy_score(train_labels, train_preds)
        avg_train_loss = sum(train_losses)/len(train_losses)

        # -------------------------------
        # Validation
        # -------------------------------
        model.eval()
        val_preds = []
        val_labels = []
        val_losses = []

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(DEVICE)
                labels = labels.float().unsqueeze(1).to(DEVICE)

                outputs = model(images)
                loss = criterion(outputs, labels)
                val_losses.append(loss.item())

                val_preds.extend((outputs.cpu() > 0.5).int().numpy())
                val_labels.extend(labels.cpu().numpy())

        val_acc = accuracy_score(val_labels, val_preds)
        avg_val_loss = sum(val_losses)/len(val_losses)

        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] | "
              f"Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.4f} | "
              f"Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.4f}")

        # -------------------------------
        # Save best model
        # -------------------------------
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            checkpoint_path = os.path.join(CHECKPOINT_DIR, "best_model.pth")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Saved checkpoint: {checkpoint_path}")

        # Step scheduler
        scheduler.step()

    print("Training complete!")
