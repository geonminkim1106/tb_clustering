import os
import pandas as pd
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
from PIL import Image

# Weak augmentation: Minimal geometric distortion tailored for Chest X-rays
weak_transform = T.Compose([
    T.Resize((224, 224)),                                 # Resize to standard input dimension
    T.RandomAffine(degrees=5, translate=(0.02, 0.02)),    # Slight rotation and translation to simulate patient positioning
    T.ToTensor()                                          # Convert PIL image to PyTorch tensor
])

# Strong augmentation: Designed to prevent the loss of peripheral lung lesions
strong_transform = T.Compose([
    T.Resize((256, 256)),                                 # Scale up slightly before cropping
    # CRITICAL: Bounding scale to (0.8, 1.0) ensures critical lung boundaries are not cropped out
    T.RandomResizedCrop(224, scale=(0.8, 1.0)),           
    T.RandomApply([T.GaussianBlur(kernel_size=3)], p=0.5), # Apply Gaussian blur with 50% probability
    T.RandomApply([T.ColorJitter(brightness=0.2, contrast=0.2)], p=0.5), # Simulate different X-ray exposures
    T.ToTensor()
])

class CXRContrastiveDataset(Dataset):
    """
    Custom Dataset for Contrastive Learning.
    Returns two augmented views (weak and strong) of the same image.
    """
    def __init__(self, dataframe, image_dir, weak_tf, strong_tf):
        self.df = dataframe.reset_index(drop=True)        # Reset index to avoid out-of-bounds errors
        self.image_dir = image_dir                        # Directory containing the raw images
        self.weak_tf = weak_tf                            # Weak transformation pipeline
        self.strong_tf = strong_tf                        # Strong transformation pipeline

    def __len__(self):
        return len(self.df)                               # Return total number of samples

    def __getitem__(self, idx):
        img_name = self.df.loc[idx, 'Image Index']        # Retrieve filename from dataframe
        img_path = os.path.join(self.image_dir, img_name) # Construct full image path

        # Convert grayscale X-ray to 3-channel RGB to match Xception's expected input shape
        image = Image.open(img_path).convert('RGB')

        # Generate two different augmented views for contrastive learning
        view_1 = self.weak_tf(image)
        view_2 = self.strong_tf(image)

        return view_1, view_2
