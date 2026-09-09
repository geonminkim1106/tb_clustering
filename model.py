import torch
import torch.nn as nn
import timm


class TCL_Xception(nn.Module):
    def __init__(self, feature_dim=128, num_clusters=512):
        super(TCL_Xception, self).__init__()

        # 1. Base Backbone: ImageNet Pretrained Xception
        # num_classes=0으로 설정하여 원본 분류기(FC Layer)를 제거하고 Global Average Pooling 피처만 추출
        self.backbone = timm.create_model('xception', pretrained=True, num_classes=0, global_pool='avg')

        # Xception의 기본 출력 차원은 2048
        backbone_out_dim = 2048

        # 2. ICH (Instance-level Contrastive Head)
        # 2048 -> 512 -> feature_dim(128) 차원으로 축소하는 비선형 프로젝션 헤드 (SimCLR 구조)
        self.ich = nn.Sequential(
            nn.Linear(backbone_out_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, feature_dim)
        )

        # 3. CCH (Cluster-level Contrastive Head)
        # 2048 -> 512 -> num_clusters 차원으로 매핑하여 군집 할당(Cluster Assignment) 확률 계산
        self.cch = nn.Sequential(
            nn.Linear(backbone_out_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_clusters)
        )

    def forward(self, x):
        # [Batch, 3, 224, 224] -> [Batch, 2048]
        representation = self.backbone(x)

        # [Batch, 2048] -> [Batch, 128]
        z_instance = self.ich(representation)

        # [Batch, 2048] -> [Batch, 512] (num_clusters)
        z_cluster = self.cch(representation)

        # L2 정규화(Normalization) - 코사인 유사도 기반 대조 학습의 필수 과정
        z_instance = nn.functional.normalize(z_instance, dim=1)
        z_cluster = nn.functional.normalize(z_cluster, dim=1)

        return z_instance, z_cluster


# ---------------------------------------------
# 모델 초기화 및 Forward Pass 테스트
# ---------------------------------------------
if __name__ == '__main__':
    # 모델 인스턴스화 (GPU 사용 가능 시 CUDA로 이동)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TCL_Xception(feature_dim=128, num_clusters=512).to(device)

    # DataLoader에서 추출했던 텐서 크기와 동일한 더미(Dummy) 데이터 생성
    dummy_input = torch.randn(32, 3, 224, 224).to(device)

    # Forward Pass 실행
    out_instance, out_cluster = model(dummy_input)

    print(f"Input Shape: {dummy_input.shape}")
    print(f"ICH Output Shape (Instance): {out_instance.shape}")
    print(f"CCH Output Shape (Cluster): {out_cluster.shape}")