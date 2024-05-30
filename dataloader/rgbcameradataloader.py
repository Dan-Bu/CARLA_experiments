import os
from PIL import Image
import torch
import torch.nn.functional as F
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
import settings

class image_sem_seg_dataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        self.image_dir = os.path.join(root_dir, 'images')
        self.mask_dir = os.path.join(root_dir, 'masks')

        self.image_filenames = os.listdir(self.image_dir)

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_dir, self.image_filenames[idx])
        mask_name = os.path.join(self.mask_dir, self.image_filenames[idx])  # Assuming the mask filename matches the image filename

        image = Image.open(img_name).convert('RGB')
        mask = Image.open(mask_name).convert('RGB')
        mask = self.colored_mask_to_label_map(mask)
        if self.transform:
            image = self.transform(image)

        return image, mask
    
    def colored_mask_to_label_map(self, mask):
        # Convert colored mask to numpy array
        #mask_np = np.array(mask)

        # Define color to class mapping
        color_to_class = {
            (0, 0, 0): 0,   
            (128, 64, 128): 1,  
            (244, 35, 232): 2, 
            (70, 70, 70): 3,   
            (102, 102, 156): 4,  
            (190, 153, 153): 5, 
            (153, 153, 153): 6,   
            (250, 170, 30): 7,  
            (220, 220, 0): 8, 
            (107, 142, 35): 9,   
            (152, 251, 152): 10,  
            (70, 130, 180): 11, 
            (220, 20, 60): 12,   
            (255, 0, 0): 13,  
            (0, 0, 142): 14, 
            (0, 0, 70): 15,   
            (0, 60, 100): 16,  
            (0, 60, 100): 17, 
            (0, 0, 230): 18,   
            (119, 11, 32): 19,  
            (110, 190, 160): 20,
            (170, 120, 50): 21,   
            (55, 90, 80): 22,  
            (45, 60, 150): 23,
            (157, 234, 50): 24,   
            (81, 0, 81): 25,  
            (150, 100, 100): 26,
            (230, 150, 140): 27,   
            (180, 165, 180): 28      
            # Add more color-class mappings as needed
        }

        if isinstance(mask, Image.Image):
            mask_np = np.array(mask)
        elif isinstance(mask, np.ndarray):
            mask_np = mask
        elif isinstance(mask, torch.Tensor):
            # Convert PyTorch tensor to NumPy array
            mask_np = mask.cpu().numpy()
        else:
            raise TypeError(f"Input type {type(mask)} is not supported. Provide a PIL Image or numpy array.")

        # Initialize an array to hold the class indices
        class_indices = np.zeros((mask_np.shape[0], mask_np.shape[1]), dtype=np.int64)

        # Map the colors to class indices
        for color, class_idx in color_to_class.items():
            mask_idx = (mask_np == color).all(axis=2)
            class_indices[mask_idx] = class_idx

        # Convert class indices to a tensor
        class_indices_tensor = torch.from_numpy(class_indices).long().unsqueeze(0)

        # One-hot encode the class indices
        one_hot_tensor = F.one_hot(class_indices_tensor, num_classes=29).permute(0, 3, 1, 2).float()

        return one_hot_tensor

    
class CustomDataLoader(pl.LightningDataModule):
    def __init__(self, root_dir, batch_size, num_workers=settings.dataloader_num_workers):
        super().__init__()
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        # Define transformations
        transform = transforms.Compose([
            transforms.Resize((settings.image_h, settings.image_w)),  # Resize images and masks to the same size
            transforms.ToTensor(),  # Convert PIL images to tensors
        ])

        # Load dataset
        self.train_dataset = image_sem_seg_dataset(root_dir=self.root_dir, transform=transform)
        self.val_dataset = image_sem_seg_dataset(root_dir=self.root_dir + "/val", transform=transform) 
        self.test_dataset = image_sem_seg_dataset(root_dir=self.root_dir + "/test", transform=transform)

        return

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)