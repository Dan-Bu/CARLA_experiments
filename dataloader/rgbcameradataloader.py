import os
from PIL import Image
import torch
from torch.utils.data import Dataset

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
        mask = Image.open(mask_name).convert('L')  # Assuming masks are grayscale

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        return image, mask