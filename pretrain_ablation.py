import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import GroupShuffleSplit

# Import custom dataset and model components
from data_loader import CXRContrastiveDataset, weak_transform, strong_transform
from model import TCL_Xception

class TCLLoss(nn.Module):
    """
    Dynamic Twin-Contrastive Loss module supporting Ablation modes (ICH, CCH, BOTH).
    """
    def __init__(self, batch_size, temperature_inst=0.5, temperature_cluster=1.0):
        super(TCLLoss, self).__init__()
        self.batch_size = batch_size
        self.temp_inst = temperature_inst                   # Temperature scaling for instance contrast
        self.temp_cluster = temperature_cluster             # Temperature scaling for cluster contrast
        self.criterion = nn.CrossEntropyLoss()

    def info_nce_loss(self, view1, view2, temperature):
        """Calculates the standard InfoNCE loss for contrastive learning."""
        features = torch.cat([view1, view2], dim=0)         # Concatenate both views [2*B, D]
        
        # Calculate cosine similarity matrix
        similarity_matrix = torch.matmul(features, features.T) / temperature
        
        # Create a mask to remove self-similarity (diagonal elements)
        mask = torch.eye(features.shape[0], dtype=torch.bool).to(features.device)
        similarity_matrix = similarity_matrix[~mask].view(features.shape[0], -1)

        # Calculate similarity between positive pairs (view1 and view2 of the same image)
        positives = torch.sum(view1 * view2, dim=-1) / temperature
        positives = torch.cat([positives, positives], dim=0).view(-1, 1)

        # Labels are always 0 because the positive pair is placed at the 0th index
        labels = torch.zeros(features.shape[0], dtype=torch.long).to(features.device)
        
        # Concatenate positives and negatives to form logits
        logits = torch.cat([positives, similarity_matrix], dim=1)
        return self.criterion(logits, labels)

    def forward(self, z_i, z_j, c_i, c_j, mode='BOTH'):
        # Calculate Instance-level loss (ICH)
        loss_instance = self.info_nce_loss(z_i, z_j, self.temp_inst)
        
        # Calculate Cluster-level loss (CCH) across the batch dimension (using transposed probabilities)
        p_i = nn.functional.softmax(c_i, dim=1)
        p_j = nn.functional.softmax(c_j, dim=1)
        loss_cluster = self.info_nce_loss(p_i.T, p_j.T, self.temp_cluster)

        # Dynamic mode switching for ablation studies
        if mode == 'ICH':
            total_loss = loss_instance
            loss_cluster_val = 0.0
            loss_inst_val = loss_instance.item()
        elif mode == 'CCH':
            total_loss = loss_cluster
            loss_inst_val = 0.0
            loss_cluster_val = loss_cluster.item()
        else: # BOTH (TCL_Full)
            total_loss = loss_instance + loss_cluster
            loss_inst_val = loss_instance.item()
            loss_cluster_val = loss_cluster.item()

        return total_loss, loss_inst_val, loss_cluster_val


if __name__ == '__main__':
    # Define paths (Adjust to your local or cloud environment)
    csv_path = "/content/drive/MyDrive/TB_data/Data_Entry_2017.csv"
    image_dir = "/content/drive/MyDrive/TB_data/images_001/images"
    save_dir = "/content/drive/MyDrive/TB_data/"

    # Hyperparameters
    batch_size = 16 
    epochs = 50 
    learning_rate = 1e-4

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load metadata and filter out missing images
    df = pd.read_csv(csv_path)
    existing_files = set(os.listdir(image_dir))
    df = df[df['Image Index'].isin(existing_files)].reset_index(drop=True)

    # Patient-level split to prevent data leakage (same patient appearing in both train and val)
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, val_idx = next(gss.split(df, groups=df['Patient ID']))
    train_df = df.iloc[train_idx]

    # Initialize Dataset and DataLoader
    actual_batch_size = min(batch_size, len(train_df))
    train_dataset = CXRContrastiveDataset(train_df, image_dir, weak_transform, strong_transform)
    train_loader = DataLoader(train_dataset, batch_size=actual_batch_size, shuffle=True, num_workers=2, drop_last=True)

    # Define ablation modes to run sequentially
    ablation_modes = ['ICH', 'CCH', 'BOTH']

    for mode in ablation_modes:
        print(f"Starting {mode}-Only Pretraining ({epochs} Epochs)")
        
        # Reinitialize model and optimizer for each mode to ensure a clean slate
        model = TCL_Xception(feature_dim=128, num_clusters=512).to(device)
        criterion = TCLLoss(batch_size=actual_batch_size).to(device)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        for epoch in range(epochs):
            model.train()
            epoch_loss = 0.0

            for batch_idx, (view_1, view_2) in enumerate(train_loader):
                # Move augmented views to GPU
                view_1, view_2 = view_1.to(device), view_2.to(device)
                optimizer.zero_grad()

                # Forward pass for both views
                z_i, c_i = model(view_1)
                z_j, c_j = model(view_2)

                # Compute loss based on the current ablation mode
                loss, loss_inst_val, loss_clust_val = criterion(z_i, z_j, c_i, c_j, mode=mode)
                
                # Backpropagation
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(train_loader)
            print(f"==> Epoch [{epoch + 1}/{epochs}] | {mode} Loss: {avg_loss:.4f}")

            # Defensive Checkpointing: Save weights every 10 epochs to prevent data loss
            if (epoch + 1) % 10 == 0:
                temp_path = os.path.join(save_dir, f"{mode.lower()}_epoch_{epoch+1}.pth")
                torch.save(model.state_dict(), temp_path)

        # Save final weights for the current mode
        save_path = os.path.join(save_dir, f"{mode.lower()}_model.pth")
        torch.save(model.state_dict(), save_path)
