import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision.transforms as T
from PIL import Image
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, confusion_matrix
from sklearn.model_selection import StratifiedShuffleSplit

from model import TCL_Xception


# ==========================================
# 1. Dataset & Model (기존과 동일)
# ==========================================
class TBClassificationDataset(Dataset):
    def __init__(self, dataframe, image_dir, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = self.df.loc[idx, 'filename']
        label = self.df.loc[idx, 'label']
        img_path = os.path.join(self.image_dir, img_name)
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)
        return image, torch.tensor([label], dtype=torch.float32)


class TBClassifier(nn.Module):
    def __init__(self, pretrained_model_path=None):
        super(TBClassifier, self).__init__()
        base_model = TCL_Xception(feature_dim=128, num_clusters=512)

        if pretrained_model_path and os.path.exists(pretrained_model_path):
            base_model.load_state_dict(torch.load(pretrained_model_path))
            print("Successfully loaded pretrained TCL weights!")

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


# ==========================================
# 2. Label Efficiency 샘플러 (핵심 추가 로직)
# ==========================================
def get_stratified_subset(dataframe, fraction, random_seed):
    """지정된 비율(fraction)만큼 정상/결핵 비율을 유지하며 데이터를 추출합니다."""
    if fraction == 1.0:
        return dataframe.copy()

    # StratifiedShuffleSplit을 사용해 클래스 불균형(Class Imbalance) 유지
    sss = StratifiedShuffleSplit(n_splits=1, train_size=fraction, random_state=random_seed)
    train_idx, _ = next(sss.split(dataframe, dataframe['label']))
    return dataframe.iloc[train_idx].reset_index(drop=True)


# ==========================================
# 3. 평가 지표 계산 함수 (AUROC, AUPRC 등)
# ==========================================
def evaluate_model(model, dataloader, criterion, device):
    model.eval()
    val_loss = 0.0
    all_labels = []
    all_probs = []
    all_preds = []

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

    # 평가 지표 계산
    try:
        auroc = roc_auc_score(all_labels, all_probs)
        auprc = average_precision_score(all_labels, all_probs)
    except ValueError:
        auroc, auprc = 0.0, 0.0  # 배치 내 단일 클래스만 있을 경우의 방어 코드

    f1 = f1_score(all_labels, all_preds, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(all_labels, all_preds, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    metrics = {
        'Loss': val_loss / len(dataloader),
        'AUROC': auroc,
        'AUPRC': auprc,
        'F1': f1,
        'Sensitivity': sensitivity,
        'Specificity': specificity
    }
    return metrics


# ==========================================
# 4. 메인 실행 및 테스트 (더미 데이터로 검증)
# ==========================================
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using Device: {device}\n")

    # 가상의 다운스트림 데이터프레임 생성 (테스트용)
    dummy_df = pd.DataFrame({
        'filename': [f'img_{i}.png' for i in range(100)],
        'label': [0] * 70 + [1] * 30  # 정상 70개, 결핵 30개 (클래스 불균형 가정)
    })

    # 실험 3: Label Efficiency Test (예: 전체 데이터의 10%만 사용, Random Seed 42)
    fraction = 0.1
    seed = 42
    train_subset_df = get_stratified_subset(dummy_df, fraction=fraction, random_seed=seed)

    print(f"--- Label Efficiency Experiment ---")
    print(f"Original Data Size: {len(dummy_df)}")
    print(f"Sampled Training Data Size ({fraction * 100}%): {len(train_subset_df)}")
    print(f"Class Distribution in Sample: \n{train_subset_df['label'].value_counts()}\n")

    # 텐서 통과를 확인하기 위한 임의의 모델 및 Loss 정의
    model = TBClassifier(pretrained_model_path=None).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # (주의) 실제 학습 전 모델 구조가 에러 없이 작동하는지 확인하는 로직입니다.
    dummy_images = torch.randn(10, 3, 224, 224).to(device)
    dummy_labels = torch.randint(0, 2, (10, 1)).float().to(device)

    # 1 Epoch 흉내내기
    model.train()
    optimizer.zero_grad()
    outputs = model(dummy_images)
    loss = criterion(outputs, dummy_labels)
    loss.backward()
    optimizer.step()

    print("[Model Check & Metrics Output Format]")
    print(f"Train Loss: {loss.item():.4f}")


    # 임의의 데이터로더를 만들었다고 가정하고 평가 함수 통과 테스트
    class DummyLoader:
        def __iter__(self):
            yield dummy_images, dummy_labels

        def __len__(self): return 1


    val_metrics = evaluate_model(model, DummyLoader(), criterion, device)
    for k, v in val_metrics.items():
        print(f"Val {k}: {v:.4f}")
