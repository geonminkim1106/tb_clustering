import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from sklearn.model_selection import StratifiedShuffleSplit
from PIL import Image
import matplotlib.pyplot as plt
 
import timm
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

def set_seed(seed=42):
    """Locks random seeds to ensure visualization targets are consistent."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

image_dir = "/content/drive/MyDrive/TB_data/Shenzhen/ChinaSet_AllFiles/ChinaSet_AllFiles/CXR_png"
w_path = "/content/drive/MyDrive/TB_data/both_model.pth"

# Load Shenzhen Dataset metadata
file_list = [f for f in os.listdir(image_dir) if f.endswith('.png')]
df_list = [{'filename': f, 'label': int(f.split('_')[-1][0])} for f in file_list]
real_df = pd.DataFrame(df_list)

# Recreate the exact same 10% data split used in evaluation
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, val_idx = next(sss.split(real_df, real_df['label']))
full_train_df = real_df.iloc[train_idx].reset_index(drop=True)
val_df = real_df.iloc[val_idx].reset_index(drop=True)

sss_sub = StratifiedShuffleSplit(n_splits=1, train_size=0.1, random_state=42)
sub_idx, _ = next(sss_sub.split(full_train_df, full_train_df['label']))
train_df = full_train_df.iloc[sub_idx].reset_index(drop=True)

class TBClassificationDataset(Dataset):
    """Dataset modified to also return the unnormalized original image for overlay visualization."""
    def __init__(self, dataframe, image_dir, transform=None):
        self.df = dataframe
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = self.df.loc[idx, 'filename']
        img_path = os.path.join(self.image_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        label = self.df.loc[idx, 'label']
        
        # Save a resized original copy for Grad-CAM overlay
        orig_image = image.resize((224, 224)) 
        
        if self.transform:
            image_tensor = self.transform(image)
        return image_tensor, torch.tensor(label, dtype=torch.float32), np.array(orig_image), img_name

train_transform = T.Compose([
    T.Resize((224, 224)), T.RandomHorizontalFlip(p=0.5),
    T.ColorJitter(brightness=0.1, contrast=0.1), T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
val_transform = T.Compose([
    T.Resize((224, 224)), T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dataset = TBClassificationDataset(train_df, image_dir, transform=train_transform)
val_dataset = TBClassificationDataset(val_df, image_dir, transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=2, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False) # Batch size 1 for targeted visualization

class TBClassifier(nn.Module):
    """Classifier structure tailored to allow Grad-CAM hooks."""
    def __init__(self, pretrained_model_path=None):
        super(TBClassifier, self).__init__()
        self.backbone = timm.create_model('xception', pretrained=False, num_classes=0)
        
        if pretrained_model_path and os.path.exists(pretrained_model_path):
            state_dict = torch.load(pretrained_model_path, map_location='cpu')
            backbone_state_dict = {k.replace('backbone.', ''): v for k, v in state_dict.items() if k.startswith('backbone.')}
            self.backbone.load_state_dict(backbone_state_dict, strict=False)
            
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(2048, 1)
        )

    def forward(self, x):
        features = self.backbone(x)
        logits = self.classifier(features)
        # Note: Do not squeeze(-1) here to maintain compatibility with pytorch_grad_cam requirements
        return logits 

model = TBClassifier(pretrained_model_path=w_path).to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Fine-tune the model rapidly to establish decision boundaries for visualization
for epoch in range(5):
    model.train()
    for images, labels, _, _ in train_loader:
        images, labels = images.to(device), labels.to(device).unsqueeze(-1)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

model.eval()

# Set the target layer for Grad-CAM. 'conv4' is typically the final block in legacy_xception.
target_layers = [model.backbone.conv4]
# Index 0 represents the singular logit output of our binary classifier
targets = [ClassifierOutputTarget(0)]
cam = GradCAM(model=model, target_layers=target_layers)

# Extract exactly 2 Normal and 2 TB samples from the validation set
sample_images = {'Normal': [], 'TB': []}
for image_tensor, label, orig_image, img_name in val_loader:
    lbl_idx = int(label.item())
    key = 'TB' if lbl_idx == 1 else 'Normal'
    if len(sample_images[key]) < 2:
        sample_images[key].append((image_tensor, orig_image.squeeze(0), img_name[0], lbl_idx))
    if len(sample_images['Normal']) == 2 and len(sample_images['TB']) == 2:
        break

# Setup 2x4 plotting grid (Row 1: Normal, Row 2: TB. Each item occupies 2 columns for Original/CAM)
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
plt.subplots_adjust(wspace=0.1, hspace=0.3)

plot_idx = 0
for key in ['Normal', 'TB']:
    for img_tensor, orig_img, img_name, true_lbl in sample_images[key]:
        input_tensor = img_tensor.to(device)
        
        # Get model prediction to display alongside true label
        with torch.no_grad():
            output = model(input_tensor)
            prob = torch.sigmoid(output).item()
            pred_lbl = 1 if prob > 0.5 else 0
            
        # Generate the heatmap
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]
        
        # Normalize original image to [0, 1] bounds for the overlay utility
        rgb_img = np.float32(orig_img) / 255.0
        cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
        
        # Safely compute row/column indices to avoid Out-Of-Bounds errors
        row, col = plot_idx // 2, (plot_idx % 2) * 2
        
        # Plot 1: Unmodified Original X-ray
        ax1 = axes[row, col]
        ax1.imshow(orig_img)
        ax1.axis('off')
        ax1.set_title(f"True: {key}\n{img_name}", fontsize=10)
        
        # Plot 2: X-ray with Grad-CAM overlay
        ax2 = axes[row, col + 1]
        ax2.imshow(cam_image)
        ax2.axis('off')
        
        # Format title text color dynamically based on correct/incorrect prediction
        pred_text = "TB" if pred_lbl == 1 else "Normal"
        color = "green" if pred_lbl == true_lbl else "red"
        ax2.set_title(f"Pred: {pred_text} ({prob:.2f})", fontsize=10, color=color)
        
        plot_idx += 1

plt.suptitle("TCL_Full (10% Data) - Grad-CAM Visualizations", fontsize=16, weight='bold')
plt.show()
