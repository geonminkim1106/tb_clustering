import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import GroupShuffleSplit

# 앞서 만든 두 개의 모듈을 불러옵니다.
from data_loader import CXRContrastiveDataset, weak_transform, strong_transform
from model import TCL_Xception


class TCLLoss(nn.Module):
    """
    Instance-level(ICH)과 Cluster-level(CCH)을 동시에 학습시키는 TCL Loss
    """

    def __init__(self, batch_size, temperature_inst=0.5, temperature_cluster=1.0):
        super(TCLLoss, self).__init__()
        self.batch_size = batch_size
        self.temp_inst = temperature_inst
        self.temp_cluster = temperature_cluster
        self.criterion = nn.CrossEntropyLoss()

    def info_nce_loss(self, view1, view2, temperature):
        # [2*Batch, Dim] 형태로 결합
        features = torch.cat([view1, view2], dim=0)

        # 코사인 유사도 행렬 계산
        similarity_matrix = torch.matmul(features, features.T) / temperature

        # 자기 자신과의 유사도(대각 원소)를 마스킹하여 제외
        mask = torch.eye(features.shape[0], dtype=torch.bool).to(features.device)
        similarity_matrix = similarity_matrix[~mask].view(features.shape[0], -1)

        # 양성 샘플(Positive pair) 간의 유사도 추출
        positives = torch.sum(view1 * view2, dim=-1) / temperature
        positives = torch.cat([positives, positives], dim=0).view(-1, 1)

        # Cross Entropy 계산을 위한 타겟 라벨 (항상 0번 인덱스가 Positive가 되도록 정렬됨)
        labels = torch.zeros(features.shape[0], dtype=torch.long).to(features.device)

        # 로짓(Logits) 구성: [Positive, Negative 1, Negative 2, ...]
        logits = torch.cat([positives, similarity_matrix], dim=1)
        return self.criterion(logits, labels)

    def forward(self, z_i, z_j, c_i, c_j):
        # 1. Instance-level Loss (배치 내 이미지 단위 대조)
        loss_instance = self.info_nce_loss(z_i, z_j, self.temp_inst)

        # 2. Cluster-level Loss (소프트맥스로 확률화 후, 클러스터 차원(Columns) 기준으로 대조)
        p_i = nn.functional.softmax(c_i, dim=1)
        p_j = nn.functional.softmax(c_j, dim=1)
        # 클러스터 차원으로 전치(Transpose)하여 군집 특성을 대조 학습
        loss_cluster = self.info_nce_loss(p_i.T, p_j.T, self.temp_cluster)

        # 3. 최종 TCL Loss = 인스턴스 손실 + 군집 손실
        total_loss = loss_instance + loss_cluster
        return total_loss, loss_instance, loss_cluster


if __name__ == '__main__':
    # ==========================================
    # 1. 설정 및 하이퍼파라미터
    # ==========================================
    csv_path = r"C:\Users\geonm\OneDrive\Desktop\Data_Entry_2017.csv"
    image_dir = r"C:\Users\geonm\Downloads\images_001\images"

    batch_size = 16  # VRAM 용량에 따라 16 또는 32로 조절
    epochs = 5  # 프로토타이핑용 테스트 에포크
    learning_rate = 1e-4

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using Device: {device}")

    # ==========================================
    # 2. 데이터 준비 (data_loader.py 로직 재활용)
    # ==========================================
    df = pd.read_csv(csv_path)
    existing_files = set(os.listdir(image_dir))
    df = df[df['Image Index'].isin(existing_files)].reset_index(drop=True)

    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, val_idx = next(gss.split(df, groups=df['Patient ID']))
    train_df = df.iloc[train_idx]

    actual_batch_size = min(batch_size, len(train_df))

    train_dataset = CXRContrastiveDataset(train_df, image_dir, weak_transform, strong_transform)
    train_loader = DataLoader(train_dataset, batch_size=actual_batch_size, shuffle=True, num_workers=4, drop_last=True)

    # ==========================================
    # 3. 모델, Loss, Optimizer 초기화
    # ==========================================
    model = TCL_Xception(feature_dim=128, num_clusters=512).to(device)
    criterion = TCLLoss(batch_size=actual_batch_size).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # ==========================================
    # 4. 학습 루프 (Train Loop)
    # ==========================================
    print("\n--- Starting TCL Pretraining ---")
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0

        for batch_idx, (view_1, view_2) in enumerate(train_loader):
            view_1, view_2 = view_1.to(device), view_2.to(device)

            optimizer.zero_grad()

            # Forward Pass: 두 개의 뷰를 모델에 통과시킴
            z_i, c_i = model(view_1)
            z_j, c_j = model(view_2)

            # Loss 계산
            loss, loss_inst, loss_clust = criterion(z_i, z_j, c_i, c_j)

            # Backpropagation & 최적화
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            print(f"Epoch [{epoch + 1}/{epochs}] Batch [{batch_idx + 1}/{len(train_loader)}] "
                  f"Loss: {loss.item():.4f} (Inst: {loss_inst.item():.4f}, Clust: {loss_clust.item():.4f})")

        print(f"==> Epoch {epoch + 1} Average Loss: {epoch_loss / len(train_loader):.4f}\n")

    print("Pretraining Prototyping Complete!")
