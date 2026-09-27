import pandas as pd
import os
from sklearn.model_selection import GroupShuffleSplit
import torchvision.transforms as T
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

df = pd.read_csv(r"C:\Users\geonm\OneDrive\Desktop\Data_Entry_2017.csv")

# 환자 ID(Patient ID)가 쪼개지지 않도록 Train/Val 분할
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, val_idx = next(gss.split(df, groups=df['Patient ID']))
train_df = df.iloc[train_idx]
val_df = df.iloc[val_idx]

# 흉부 X-ray 특성을 고려한 변형 (과도한 기하학적 왜곡 배제)
weak_transform = T.Compose([
    T.Resize((224, 224)),
    T.RandomAffine(degrees=5, translate=(0.02, 0.02)),
    T.ToTensor()
])

strong_transform = T.Compose([
    T.Resize((256, 256)),
    # 폐 가장자리(Peripheral) 병변 보존을 위해 scale 하한선을 0.8로 방어
    T.RandomResizedCrop(224, scale=(0.8, 1.0)),
    T.RandomApply([T.GaussianBlur(kernel_size=3)], p=0.5),
    T.RandomApply([T.ColorJitter(brightness=0.2, contrast=0.2)], p=0.5),
    T.ToTensor()
])

class CXRContrastiveDataset(Dataset):
    def __init__(self, dataframe, image_dir, weak_tf, strong_tf):
        self.df = dataframe.reset_index(drop=True)
        self.image_dir = image_dir
        self.weak_tf = weak_tf
        self.strong_tf = strong_tf

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = self.df.loc[idx, 'Image Index']
        img_path = os.path.join(self.image_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        view_1 = self.weak_tf(image)
        view_2 = self.strong_tf(image)
        return view_1, view_2

def visualize_views(view1_batch, view2_batch, num_samples=4):
    fig, axes = plt.subplots(num_samples, 2, figsize=(8, 4 * num_samples))
    for i in range(num_samples):
        img1 = view1_batch[i].permute(1, 2, 0).numpy()
        img2 = view2_batch[i].permute(1, 2, 0).numpy()

        axes[i, 0].imshow(img1)
        axes[i, 0].set_title(f"Sample {i+1}: Weak Aug")
        axes[i, 0].axis('off')

        axes[i, 1].imshow(img2)
        axes[i, 1].set_title(f"Sample {i+1}: Strong Aug")
        axes[i, 1].axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    image_dir = r"C:\Users\geonm\Downloads\images_001\images"
    existing_files = set(os.listdir(image_dir))
    filtered_train_df = train_df[train_df['Image Index'].isin(existing_files)].reset_index(drop=True)

    if len(filtered_train_df) > 0:
        batch_size = min(32, len(filtered_train_df))
        train_dataset = CXRContrastiveDataset(
            dataframe=filtered_train_df,
            image_dir=image_dir,
            weak_tf=weak_transform,
            strong_tf=strong_transform
        )
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        view_1_batch, view_2_batch = next(iter(train_loader))
        visualize_views(view_1_batch, view_2_batch, num_samples=min(4, batch_size))
