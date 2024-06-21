
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from torchvision.transforms import v2 as transforms
from torch.utils.data import DataLoader
from torchmetrics.classification import MulticlassAccuracy
import torchmetrics
import dataloader.rgbcameradataloader as customDataset
import networks.camera.unet as unet
import settings

class CustomDataModule(pl.LightningDataModule):
    def __init__(self, root_dir, batch_size):
        super().__init__()
        self.root_dir = root_dir
        self.batch_size = batch_size
        
    def setup(self, stage=None):
        
        augment_flip = 0.5
        augment_randomerase = 0.3
        self.transform_train = transforms.Compose([   #Makes sure our images are always the correct size and do data
            transforms.Resize((settings.image_h, settings.image_w)),
            transforms.RandomHorizontalFlip(p=augment_flip), # Flip data augmentation
            transforms.RandomErasing(p = augment_randomerase),
            transforms.RandAugment(),
            transforms.ToTensor(),
        ])

        augment_flip = 0.0
        augment_randomerase = 0.0
        self.transform_test = transforms.Compose([   #Makes sure our images are always the correct size and do data
            transforms.Resize((settings.image_h, settings.image_w)),
            transforms.ToTensor(),
        ])
        if stage == "fit":
            self.dataset = customDataset.image_sem_seg_dataset(root_dir=self.root_dir, transform=self.transform_train)
        else:
            self.dataset = customDataset.image_sem_seg_dataset(root_dir=self.root_dir, transform=self.transform_test)

    def train_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)

# Define the LightningModule
class LightningUNet(pl.LightningModule):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.model = unet.UNet(in_channels, out_channels)
        self.train_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=29)
        self.val_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=29)
        self.test_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=29)
    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y_true = batch
        y_pred = self(x)
        y_true = y_true.squeeze(1)  # Remove unnecessary dimension
        loss = F.cross_entropy(y_pred, y_true)
        preds = torch.argmax(y_pred, dim=1)
        true = torch.argmax(y_true, dim=1)
        self.train_accuracy.update(preds, true)
        self.log('train_loss', loss)
        self.log('train_accuracy', self.train_accuracy, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y_true = batch
        y_pred = self(x)
        y_true = y_true.squeeze(1)
        val_loss = F.cross_entropy(y_pred, y_true)
        preds = torch.argmax(y_pred, dim=1)
        true = torch.argmax(y_true, dim=1)
        self.val_accuracy.update(preds, true)
        self.log('val_loss', val_loss)
        self.log('val_accuracy', self.val_accuracy, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return val_loss

    def test_step(self, batch, batch_idx):
        x, y_true = batch
        y_pred = self(x)
        y_true = y_true.squeeze(1)
        test_loss = F.cross_entropy(y_pred, y_true)
        preds = torch.argmax(y_pred, dim=1)
        true = torch.argmax(y_true, dim=1)
        self.test_accuracy.update(preds, true)
        self.log('test_loss', test_loss)
        self.log('test_accuracy', self.test_accuracy, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return test_loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3, weight_decay=1e-5)
