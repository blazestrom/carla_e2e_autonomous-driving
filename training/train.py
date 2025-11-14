# training/train.py
import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from training.dataset import SteeringDataset, get_transforms
from training.model import PilotNet
from tqdm import tqdm
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', default='steering_labels.csv', help='CSV path (image,steering)')
    parser.add_argument('--img_root', default='.', help='root to resolve image paths from csv')
    parser.add_argument('--epochs', type=int, default=25)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--save_dir', default='../models')
    parser.add_argument('--resize_h', type=int, default=160)  # height
    parser.add_argument('--resize_w', type=int, default=320)  # width
    parser.add_argument('--val_split', type=float, default=0.1)
    parser.add_argument('--num_workers', type=int, default=4)
    return parser.parse_args()

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    for imgs, targets in tqdm(loader, desc='train', leave=False):
        imgs = imgs.to(device)
        targets = targets.to(device)
        preds = model(imgs)
        loss = criterion(preds, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * imgs.size(0)
    return running_loss / len(loader.dataset)

def eval_one_epoch(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    preds_all = []
    targets_all = []
    with torch.no_grad():
        for imgs, targets in tqdm(loader, desc='val', leave=False):
            imgs = imgs.to(device)
            targets = targets.to(device)
            preds = model(imgs)
            loss = criterion(preds, targets)
            running_loss += loss.item() * imgs.size(0)
            preds_all.append(preds.cpu().numpy())
            targets_all.append(targets.cpu().numpy())
    preds_all = np.concatenate(preds_all)
    targets_all = np.concatenate(targets_all)
    return running_loss / len(loader.dataset), preds_all, targets_all

def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # dataset + split
    transform_train = get_transforms(train=True, resize=(args.resize_h, args.resize_w))
    transform_val = get_transforms(train=False, resize=(args.resize_h, args.resize_w))
    full_ds = SteeringDataset(args.csv, args.img_root, transform=transform_train)
    n_val = int(len(full_ds) * args.val_split)
    n_train = len(full_ds) - n_val
    train_ds, val_ds = random_split(full_ds, [n_train, n_val], generator=torch.Generator().manual_seed(42))
    # fix val transforms (replace with val transform)
    val_ds.dataset.transform = transform_val

    print(f"Samples: train={len(train_ds)} val={len(val_ds)} total={len(full_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    model = PilotNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    best_val = 1e9
    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}")
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, preds, targets = eval_one_epoch(model, val_loader, criterion, device)

        print(f"  train_loss={train_loss:.6f}  val_loss={val_loss:.6f}")

        # save best
        if val_loss < best_val:
            best_val = val_loss
            save_path = os.path.join(args.save_dir, 'pilotnet_best.pth')
            torch.save({'epoch': epoch+1, 'model_state': model.state_dict(), 'optimizer_state': optimizer.state_dict()}, save_path)
            print("  Saved best model:", save_path)

    print("Training finished. Best val:", best_val)

if __name__ == '__main__':
    main()
