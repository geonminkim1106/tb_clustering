import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import GroupShuffleSplit
from data_loader import CXRContrastiveDataset, weak_transform, strong_transform
from model import TCL_Xception

class TCLLoss(nn.Module):
    def __init__(self, batch_size, temperature_inst=0.5, temperature_cluster=1.0):
        super(TCLLoss, self).__init__()
        self.batch_size = batch_size
        self.temp_inst = temperature_inst
        self.temp_cluster = temperature_cluster
        self.criterion = nn.CrossEntropyLoss()

    def info_nce_loss(self, view1, view2, temperature):
        features = torch.cat([view1, view2], dim=0)
        similarity_matrix = torch.matmul(features, features.T) / temperature
        mask = torch.eye(features.shape[0], dtype=torch.bool).to(features.device)
        similarity_matrix = similarity_matrix[~mask].view(features.shape[0], -1)

        positives = torch.sum(view1 * view2, dim=-1) / temperature
        positives = torch.cat([positives, positives], dim=0).view(-1, 1)

        labels = torch.zeros(features.shape[0], dtype=torch.long).to(features.device)
        logits = torch.cat([positives, similarity_matrix], dim=1)
        return self.criterion(logits, labels)

    def forward(self, z_i, z_j, c_i, c_j):
        loss_instance = self.info_nce_loss(z_i, z_j, self.temp_inst)
        p_i = nn.functional.softmax(c_i, dim=1)
        p_j = nn.functional.softmax(c_j, dim=1)
        loss_cluster = self.info_nce_loss(p_i.T, p_j.T, self.temp_cluster)
        total_loss = loss_instance + loss_cluster
        return total_loss, loss_instance, loss_cluster

if __name__ == '__main__':
    csv_path = r"C:\Users\geonm\OneDrive\Desktop\Data_Entry_2017.csv"
    image_dir = r"C:\Users\geonm\Downloads\images_001\images"
    batch_size = 16 
    epochs = 5 
    learning_rate = 1e-4
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    df = pd.read_csv(csv_path)
    existing_files = set(os.listdir(image_dir))
    df = df[df['Image Index'].isin(existing_files)].reset_index(drop=True)

    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, val_idx = next(gss.split(df, groups=df['Patient ID']))
    train_df = df.iloc[train_idx]
    actual_batch_size = min(batch_size, len(train_df))

    train_dataset = CXRContrastiveDataset(train_df, image_dir, weak_transform, strong_transform)
    train_loader = DataLoader(train_dataset, batch_size=actual_batch_size, shuffle=True, num_workers=4, drop_last=True)

    model = TCL_Xception(feature_dim=128, num_clusters=512).to(device)
    criterion = TCLLoss(batch_size=actual_batch_size).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for batch_idx, (view_1, view_2) in enumerate(train_loader):
            view_1, view_2 = view_1.to(device), view_2.to(device)
            optimizer.zero_grad()
            z_i, c_i = model(view_1)
            z_j, c_j = model(view_2)
            loss, loss_inst, loss_clust = criterion(z_i, z_j, c_i, c_j)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            print(f"Epoch [{epoch + 1}/{epochs}] Batch [{batch_idx + 1}/{len(train_loader)}] Loss: {loss.item():.4f}")
