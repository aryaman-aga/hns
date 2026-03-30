"""
MedMNIST v2 Dataset Loader
──────────────────────────
PneumoniaMNIST : Pediatric Chest X-Ray  — Normal (0) vs Pneumonia (1)
BreastMNIST    : Breast Ultrasound      — Benign  (0) vs Malignant (1)

Both are grayscale (1-channel), originally 28×28, upscaled to 224×224 here.
Data downloads automatically on first use via the medmnist package.
"""

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
import medmnist
from medmnist import PneumoniaMNIST, BreastMNIST, ChestMNIST
import albumentations as A
from albumentations.pytorch import ToTensorV2

IMG_SIZE = 112

TASKS = {
    'pneumonia': PneumoniaMNIST,
    'breast':    BreastMNIST,
    'chest14':   ChestMNIST,
}

CHEST14_CLASSES = [
    'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration',
    'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax',
    'Consolidation', 'Edema', 'Emphysema', 'Fibrosis',
    'Pleural Thickening', 'Hernia',
]

TASK_INFO = {
    'pneumonia': {
        'classes':     ['Normal', 'Pneumonia'],
        'n_classes':   2,
        'multi_label': False,
        'description': 'Pediatric Chest X-Ray — Pneumonia Detection',
    },
    'breast': {
        'classes':     ['Benign', 'Malignant'],
        'n_classes':   2,
        'multi_label': False,
        'description': 'Breast Ultrasound — Malignancy Detection',
    },
    'chest14': {
        'classes':     CHEST14_CLASSES,
        'n_classes':   14,
        'multi_label': True,
        'description': 'Chest X-Ray — 14 Disease Multi-Label Detection (NIH)',
    },
}


def get_transforms(split: str) -> A.Compose:
    if split == 'train':
        return A.Compose([
            A.Resize(IMG_SIZE, IMG_SIZE),
            A.HorizontalFlip(p=0.5),
            A.Affine(translate_percent=0.05, scale=(0.9, 1.1),
                 rotate=(-10, 10), p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2,
                                       contrast_limit=0.2, p=0.4),
            A.GaussianBlur(blur_limit=(3, 5), p=0.2),
            A.CoarseDropout(num_holes_range=(1, 4),
                            hole_height_range=(10, 20),
                            hole_width_range=(10, 20), p=0.1),
            A.Normalize(mean=[0.5], std=[0.5]),
            ToTensorV2(),
        ])
    return A.Compose([
        A.Resize(IMG_SIZE, IMG_SIZE),
        A.Normalize(mean=[0.5], std=[0.5]),
        ToTensorV2(),
    ])


class MedMNISTWrapper(torch.utils.data.Dataset):
    """Wraps a MedMNIST split to apply albumentations transforms."""

    def __init__(self, base_dataset, transform=None, multi_label: bool = False):
        self.dataset     = base_dataset
        self.transform   = transform
        self.multi_label = multi_label

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        img   = np.array(img)
        if img.ndim == 3:
            img = img[:, :, 0]          # (H, W, 1) → (H, W)

        if self.transform:
            img = self.transform(image=img)['image']   # (1, H, W)
        else:
            img = torch.tensor(img, dtype=torch.float32).unsqueeze(0) / 255.0

        if self.multi_label:
            # label shape (14,) — already 0/1 binary per class
            lbl = torch.tensor(label.squeeze().astype('float32'), dtype=torch.float32)
        else:
            lbl = torch.tensor(int(label.squeeze()), dtype=torch.long)

        return img, lbl


def get_dataloaders(
    task: str = 'pneumonia',
    batch_size: int = 32,
    data_dir: str = './data',
    num_workers: int = 0,
):
    """
    Downloads (if needed) and returns (train_loader, val_loader, test_loader,
    class_weights).
    """
    import os
    os.makedirs(data_dir, exist_ok=True)

    DatasetClass = TASKS[task]
    kw = dict(download=True, root=data_dir, size=28)
    multi_label  = TASK_INFO[task].get('multi_label', False)

    raw_train = DatasetClass(split='train', **kw)
    raw_val   = DatasetClass(split='val',   **kw)
    raw_test  = DatasetClass(split='test',  **kw)

    train_ds = MedMNISTWrapper(raw_train, get_transforms('train'), multi_label)
    val_ds   = MedMNISTWrapper(raw_val,   get_transforms('val'),   multi_label)
    test_ds  = MedMNISTWrapper(raw_test,  get_transforms('test'),  multi_label)

    if multi_label:
        # For multi-label use simple shuffle — per-class pos_weight handled in loss
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=False,
        )
        class_weights = None
    else:
        # Class-balanced sampler to handle label imbalance
        labels = [int(raw_train[i][1].squeeze()) for i in range(len(raw_train))]
        counts = np.bincount(labels)
        class_weights = torch.tensor(
            len(labels) / (len(counts) * counts), dtype=torch.float32
        )
        sample_weights = class_weights[labels]
        sampler = WeightedRandomSampler(
            sample_weights, len(sample_weights), replacement=True
        )
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, sampler=sampler,
            num_workers=num_workers, pin_memory=False,
        )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=False,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=False,
    )

    return train_loader, val_loader, test_loader, class_weights
