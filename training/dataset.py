# training/dataset.py
import os
import csv
from PIL import Image
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset

class SteeringDataset(Dataset):
    def __init__(self, csv_file, img_root, transform=None):
        """
        csv_file: path to steering_labels.csv (columns: image, steering)
        img_root: root directory used to resolve image paths in CSV (can be project root)
        transform: albumentations transform (on numpy RGB)
        """
        self.img_root = img_root
        self.samples = []
        with open(csv_file, newline='') as f:
            reader = csv.reader(f)
            header = next(reader)  # skip header
            for row in reader:
                if len(row) < 2:
                    continue
                img_path, steering = row[0], row[1]
                # If CSV contains full path, keep it; otherwise join with img_root
                if not os.path.isabs(img_path):
                    img_path = os.path.join(img_root, img_path)
                self.samples.append((img_path, float(steering)))
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, steering = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        img = np.array(img)

        if self.transform is not None:
            augmented = self.transform(image=img)
            img = augmented['image']
        else:
            # convert to tensor-like float32 CHW in range [-1,1]
            img = img.astype("float32") / 127.5 - 1.0
            img = np.transpose(img, (2,0,1))

        # steering -> float32
        return img, np.float32(steering)

def get_transforms(train=True, resize=(160,320)):
    h, w = resize[0], resize[1]  # (height, width)
    if train:
        return A.Compose([
            A.Crop(int(h*0.0), 0, h, w, p=1.0),   # optionally crop; here kept full
            A.Resize(h, w),
            A.RandomBrightnessContrast(p=0.5),
            A.MotionBlur(p=0.1),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=5, p=0.5),
            A.HorizontalFlip(p=0.5),
            A.Normalize(mean=(0.0,0.0,0.0), std=(1.0,1.0,1.0), max_pixel_value=127.5),
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.Resize(h, w),
            A.Normalize(mean=(0.0,0.0,0.0), std=(1.0,1.0,1.0), max_pixel_value=127.5),
            ToTensorV2(),
        ])
