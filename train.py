# train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda import is_available
from tqdm import tqdm
import os
from dataset_loader import make_loaders
from model import get_model

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for imgs, labels in tqdm(loader, desc='Train', leave=False):
        imgs = imgs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * imgs.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += imgs.size(0)
    return running_loss/total, correct/total

def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc='Val', leave=False):
            imgs = imgs.to(device)
            labels = labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * imgs.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += imgs.size(0)
    return running_loss/total, correct/total

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='./dataset')
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--save_dir', type=str, default='./checkpoints')
    parser.add_argument('--img_size', type=int, default=224)
    args = parser.parse_args()

    device = 'cuda' if is_available() else 'cpu'
    os.makedirs(args.save_dir, exist_ok=True)

    train_loader, val_loader, class_to_idx = make_loaders(args.dataset, batch_size=args.batch, img_size=args.img_size)
    model = get_model(num_classes=len(class_to_idx), pretrained=True, device=device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    best_val_acc = 0.0

    for epoch in range(1, args.epochs+1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        print(f"Epoch {epoch}/{args.epochs} | Train loss {train_loss:.4f} acc {train_acc:.4f} | Val loss {val_loss:.4f} acc {val_acc:.4f}")
        # save checkpoint
        ckpt_path = os.path.join(args.save_dir, f'epoch_{epoch:02d}.pt')
        torch.save({'epoch':epoch, 'model_state':model.state_dict(), 'class_to_idx':class_to_idx}, ckpt_path)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({'model_state':model.state_dict(), 'class_to_idx':class_to_idx}, os.path.join(args.save_dir, 'best_model.pt'))
            print("Saved best_model.pt")

if __name__ == '__main__':
    main()
