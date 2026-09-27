import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, confusion_matrix
from sklearn.model_selection import StratifiedShuffleSplit
from model import TCL_Xception

class TBClassifier(nn.Module):
    def __init__(self, pretrained_model_path=None):
        super(TBClassifier, self).__init__()
        base_model = TCL_Xception(feature_dim=128, num_clusters=512)
        self.backbone = base_model.backbone
        self.classifier = nn.Sequential(
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 1)
        )

    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)

def get_stratified_subset(dataframe, fraction, random_seed):
    if fraction == 1.0: return dataframe.copy()
    sss = StratifiedShuffleSplit(n_splits=1, train_size=fraction, random_state=random_seed)
    train_idx, _ = next(sss.split(dataframe, dataframe['label']))
    return dataframe.iloc[train_idx].reset_index(drop=True)

def evaluate_model(model, dataloader, criterion, device):
    model.eval()
    val_loss = 0.0
    all_labels, all_probs, all_preds = [], [], []

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)
            val_loss += loss.item()

            probs = torch.sigmoid(logits).cpu().numpy()
            preds = (probs > 0.5).astype(int)

            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs)
            all_preds.extend(preds)

    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    all_preds = np.array(all_preds)

    try:
        auroc = roc_auc_score(all_labels, all_probs)
        auprc = average_precision_score(all_labels, all_probs)
    except ValueError:
        auroc, auprc = 0.0, 0.0

    f1 = f1_score(all_labels, all_preds, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(all_labels, all_preds, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return {'Loss': val_loss / len(dataloader), 'AUROC': auroc, 'AUPRC': auprc, 'F1': f1}

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    dummy_df = pd.DataFrame({
        'filename': [f'img_{i}.png' for i in range(100)],
        'label': [0] * 70 + [1] * 30 
    })

    fraction, seed = 0.1, 42
    train_subset_df = get_stratified_subset(dummy_df, fraction=fraction, random_seed=seed)

    model = TBClassifier(pretrained_model_path=None).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    dummy_images = torch.randn(10, 3, 224, 224).to(device)
    dummy_labels = torch.randint(0, 2, (10, 1)).float().to(device)

    model.train()
    optimizer.zero_grad()
    outputs = model(dummy_images)
    loss = criterion(outputs, dummy_labels)
    loss.backward()
    optimizer.step()

    class DummyLoader:
        def __iter__(self): yield dummy_images, dummy_labels
        def __len__(self): return 1

    val_metrics = evaluate_model(model, DummyLoader(), criterion, device)
    for k, v in val_metrics.items(): print(f"Val {k}: {v:.4f}")
