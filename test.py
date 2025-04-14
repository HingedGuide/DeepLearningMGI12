import zipfile
import os
import shutil
from itertools import groupby
import pandas as pd
import numpy as np
import torchvision

from torchgeo.datasets import VisionDataset
import pandas as pd
from PIL import Image
import torch
import os

class UCMercedMultiLabelDataset(VisionDataset):
    def __init__(self, root, csv_file, transforms=None):
        super().__init__(root=root, transforms=transforms)
        self.annotations = pd.read_csv(csv_file)
        self.image_paths = self.annotations['filename'].tolist()
        self.labels = self.annotations.drop(columns=['filename']).values.astype(float)

    def __getitem__(self, index):
        img_path = os.path.join(self.root, self.image_paths[index])
        image = Image.open(img_path).convert('RGB')
        label = torch.tensor(self.labels[index], dtype=torch.float32)
        if self.transforms:
            image = self.transforms(image)
        return {"image": image, "label": label}

    def __len__(self):
        return len(self.annotations)

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
    root="path/to/UCMerced_LandUse",
    csv_file="path/to/labels.csv",
    transforms=transform
)

dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

model = UCMercedMultiLabelClassifier(num_classes=17)
trainer = pl.Trainer(max_epochs=10)
trainer.fit(model, dataloader)