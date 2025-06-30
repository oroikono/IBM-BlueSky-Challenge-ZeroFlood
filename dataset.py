import terratorch

from terratorch.tasks import SemanticSegmentationTask
import torch
import torchmetrics
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import albumentations
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from datasets import load_dataset


import torch
import numpy as np
from PIL import Image
from pathlib import Path
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset

import gc
from collections import Counter

MEAN = [0.31677871, 0.35971203, 0.29563584]
STD = [0.1368559, 0.10402445, 0.09053548]


class FloodDataset(Dataset):
    def __init__(
        self,
        local_path: str,
        split: str = 'train',
        target_size: tuple[int, int] = (256, 256),
        transform: A.Compose | None = None
    ):
        self.local_path = Path(local_path)
        self.split = split
        self.target_size = target_size
        self.transform = transform

        # Discover all image ↔ CSV pairs
        img_dir = self.local_path / self.split / 'image'
        img_paths = sorted(img_dir.glob('*.png'))
        csv_paths = list(self.local_path.joinpath(self.split).rglob('*.csv'))
        csv_paths = sorted(csv_paths)
        
        self.samples: list[tuple[Path, Path]] = []
        for img_path in img_paths:
            match = next(
                (c for c in csv_paths
                 if c.stem in img_path.stem or img_path.stem in c.stem),
                None
            )
            if match:
                self.samples.append((img_path, match))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        img_path, csv_path = self.samples[idx]
        # print(img_path, csv_path)
        # Load image and resizing
        image = Image.open(img_path).convert('RGB')
        image = image.resize((self.target_size[1], self.target_size[0]), Image.NEAREST)
        image = np.array(image, dtype=np.float32)
        
        # Load mask and resizing
        mask_cv = pd.read_csv(csv_path, header=None).values
        mask_arr = np.array(mask_cv)
        mask_arr = ((mask_arr == 1)).astype(np.uint8)
        if image.shape[:2] != mask_arr.shape:
            mask_pil = Image.fromarray(mask_arr, mode='L')
            mask_pil = mask_pil.resize((self.target_size[1], self.target_size[0]), Image.NEAREST)
            mask_arr = np.array(mask_pil)

        data = self.transform(image=image, mask=mask_arr)

        return { "image": data["image"], "mask": data["mask"].long()}

    def plot(self, idx: int, show_original_mask: bool = True):
        """Plot an image with a mask"""

class FloodDataModule(pl.LightningDataModule):
    def __init__(self, local_path, batch_size=4, num_workers=2):
        super().__init__()
        self.local_path = local_path
        self.batch_size = batch_size
        self.num_workers = num_workers

        # Define transforms
        self.train_transform = A.Compose([
            A.RandomRotate90(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomBrightnessContrast( brightness_limit=0.2, contrast_limit=0.2, p=0.3),
            A.Normalize( mean=MEAN, std=STD, max_pixel_value=255.0),
            ToTensorV2(),
        ])

        self.val_transform = A.Compose([
            A.Normalize( mean=MEAN, std=STD, max_pixel_value=255.0),
            ToTensorV2(),
        ])

        self.test_transform = A.Compose([
            A.Normalize( mean=MEAN, std=STD, max_pixel_value=255.0),
            ToTensorV2(),
        ])

    def setup(self, stage=None):
        
        if stage in ["fit"] or stage is None:
            self.train_dataset = FloodDataset(
                self.local_path,
                split='train',
                transform=self.train_transform
            )
            print(f"Training samples: {len(self.train_dataset)}")
        if stage in ["fit", "validate"]:
            self.val_dataset = FloodDataset(
                self.local_path,
                split='val',
                transform=self.val_transform
            )
            print(f"Validation samples: {len(self.val_dataset)}")
        if stage in ["test"]:
            self.test_dataset = FloodDataset(
                self.local_path,
                split='test',
                transform=self.test_transform
            )
            print(f"Validation samples: {len(self.test_dataset)}")

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )


