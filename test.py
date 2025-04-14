import zipfile
import os
import shutil
from itertools import groupby
import pandas as pd
import numpy as np
import torchvision

from torchvision.datasets import VisionDataset
import pandas as pd
from PIL import Image
import torch
import os



class UCMercedMultiLabelDatasetFromTxt(VisionDataset):
    def __init__(self, root, txt_file, transforms=None):
        super().__init__(root=root, transforms=transforms)
        self.image_paths = []
        self.labels = []

        with open(txt_file, 'r') as f:
            for line in f:
                parts = line.strip().split()  # split on spaces
                self.image_paths.append(parts[0])
                label = list(map(float, parts[1:]))  # convert labels to floats
                self.labels.append(label)

    def __getitem__(self, index):
        img_path = os.path.join(self.root, self.image_paths[index])
        image = Image.open(img_path).convert("RGB")
        label = torch.tensor(self.labels[index], dtype=torch.float32)
        if self.transforms:
            image = self.transforms(image)
        return {"image": image, "label": label}

    def __len__(self):
        return len(self.image_paths)

import torch.nn as nn
import pytorch_lightning as pl
from torchvision import models

class UCMercedMultiLabelClassifier(pl.LightningModule):
        def __init__(self, num_classes):
            super().__init__()
            self.model = models.resnet18(pretrained=True)
            self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
            self.loss_fn = nn.BCEWithLogitsLoss()

        def forward(self, x):
            return self.model(x)

        def training_step(self, batch, batch_idx):
            x, y = batch["image"], batch["label"]
            logits = self(x)
            loss = self.loss_fn(logits, y)
            self.log("train_loss", loss)
            return loss

        def configure_optimizers(self):
            return torch.optim.Adam(self.parameters(), lr=1e-3)

from torchvision import transforms
from torch.utils.data import DataLoader

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

dataset = UCMercedMultiLabelDataset(
    root="Images",
    txt_file="LandUse_Multilabeled.txt",
    transforms=transform
)

dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

model = UCMercedMultiLabelClassifier(num_classes=17)
trainer = pl.Trainer(max_epochs=10)
trainer.fit(model, dataloader)