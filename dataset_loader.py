# dataset_loader.py
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

class FramesDataset(Dataset):
    def __init__(self, root_dir, split='train', img_size=224, transform=None):
        self.root_dir = root_dir
        self.img_size = img_size
        self.samples = []  # list of (path, label)
        classes = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
        self.class_to_idx = {c:i for i,c in enumerate(classes)}
        for c in classes:
            folder = os.path.join(root_dir, c)
            for f in os.listdir(folder):
                if f.lower().endswith(('.png','.jpg','.jpeg')):
                    self.samples.append((os.path.join(folder, f), self.class_to_idx[c]))
        self.transform = transform or T.Compose([
            T.Resize((img_size, img_size)),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert('RGB')
        img = self.transform(img)
        return img, label

def make_loaders(dataset_dir, batch_size=32, img_size=224, num_workers=4):
    # Простейшее разбиение: случайно 80/20
    from sklearn.model_selection import train_test_split
    import random
    dataset = FramesDataset(dataset_dir, img_size=img_size)
    indices = list(range(len(dataset)))
    train_idx, val_idx = train_test_split(indices, test_size=0.2, random_state=42, stratify=[l for _,l in dataset.samples])
    # Subset loaders
    from torch.utils.data import Subset
    train_loader = DataLoader(Subset(dataset, train_idx), batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(Subset(dataset, val_idx), batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, dataset.class_to_idx
