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

from torch.utils.checkpoint import checkpoint
import torch.nn as nn

class FloodRiskModel(pl.LightningModule):
    def __init__(self, num_classes=2, tim=["S1RTC","DEM","LULC","NDVI"]):
        super().__init__()
        self.num_classes = num_classes

        # TerraTorch model configuration for flood segmentation
        model_args = {
            "backbone": "terramind_v1_base_tim",
            "backbone_pretrained": True,
            "backbone_modalities": ["RGB"],
            "backbone_tim_modalities": tim,
            "necks": [
                {"name": "SelectIndices", "indices": [0, 1, 2, 3]},
                {"name": "ReshapeTokensToImage", "remove_cls_token": False},
                {"name": "LearnedInterpolateToPyramidal"}
            ],
            "decoder": "UNetDecoder",
            "decoder_channels": [512, 256, 128, 64],
            "num_classes": num_classes,
        }

        self.model = SemanticSegmentationTask(
            model_factory="EncoderDecoderFactory",
            model_args=model_args,
            loss="dice",
            optimizer="AdamW",
            lr=2e-5,
            ignore_index=-1,
            freeze_backbone=False,
            freeze_decoder=False,
            plot_on_val=False,

            class_names=["SafetyZone", "RiskyZone"],
        )

        # Metrics
        self.train_iou = torchmetrics.JaccardIndex(task="multiclass", num_classes=num_classes, ignore_index=-1)
        self.val_iou = torchmetrics.JaccardIndex(task="multiclass", num_classes=num_classes, ignore_index=-1)
        self.train_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes, ignore_index=-1)
        self.val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes, ignore_index=-1)
        self.train_f1 = torchmetrics.F1Score(task="multiclass", num_classes=num_classes, ignore_index=-1)
        self.val_f1 = torchmetrics.F1Score(task="multiclass", num_classes=num_classes, ignore_index=-1)

    def forward(self, x):
        return self.model({"RGB": x})

    def training_step(self, batch, batch_idx):
        images, masks = batch["image"], batch["mask"]

        outputs = self.model({"RGB": images})
        logits = outputs.output

        # Cross-entropy loss
        loss = torch.nn.functional.cross_entropy(logits, masks)
        preds = torch.argmax(logits, dim=1)

        self.train_iou(preds, masks)
        self.train_acc(preds, masks)
        self.train_f1(preds, masks)

        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_iou', self.train_iou, on_epoch=True, prog_bar=True)
        self.log('train_acc', self.train_acc, on_epoch=True, prog_bar=True)
        self.log('train_f1', self.train_f1, on_epoch=True, prog_bar=True)
        
        #  Add periodic cleanup:
        if batch_idx % 20 == 0:  # Every 20 batches
            del outputs, logits, preds  # Clean intermediate variables
            torch.cuda.empty_cache()

        return loss

    def validation_step(self, batch, batch_idx):
        images, masks = batch["image"], batch["mask"]

        outputs = self.model({"RGB": images})
        logits = outputs.output

        loss = torch.nn.functional.cross_entropy(logits, masks)
        preds = torch.argmax(logits, dim=1)

        self.val_iou(preds, masks)
        self.val_acc(preds, masks)
        self.val_f1(preds, masks)

        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        self.log('val_iou', self.val_iou, on_epoch=True, prog_bar=True)
        self.log('val_acc', self.val_acc, on_epoch=True, prog_bar=True)
        self.log('val_f1', self.val_f1, on_epoch=True, prog_bar=True)

        # Add cleanup for validation too:
        del outputs, logits, preds
        if batch_idx % 10 == 0:
            torch.cuda.empty_cache()

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=2e-5, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
        return [optimizer], [scheduler]
