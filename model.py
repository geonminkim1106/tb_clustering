import torch
import torch.nn as nn
import timm

class TCL_Xception(nn.Module):
    """
    Twin-Contrastive Learning (TCL) Architecture using Xception backbone.
    Features dual projection heads for Instance and Cluster-level learning.
    """
    def __init__(self, feature_dim=128, num_clusters=512):
        super(TCL_Xception, self).__init__()
        
        # Load Xception backbone without pretrained weights, discarding the final classification layer
        self.backbone = timm.create_model('xception', pretrained=False, num_classes=0)
        
        # Instance-level Contrastive Head (ICH): Maps features to a lower-dimensional embedding space
        self.instance_head = nn.Sequential(
            nn.Linear(2048, 512),                         # Reduce from backbone output (2048) to 512
            nn.BatchNorm1d(512),                          # Normalize for training stability
            nn.ReLU(),                                    # Non-linear activation
            nn.Linear(512, feature_dim)                   # Final instance feature dimension (e.g., 128)
        )
        
        # Cluster-level Contrastive Head (CCH): Maps features to cluster assignment probabilities
        self.cluster_head = nn.Sequential(
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, num_clusters)                  # Final dimension represents the number of semantic clusters
        )

    def forward(self, x):
        # Extract base representations from the backbone
        features = self.backbone(x)
        
        # Pass features through both heads simultaneously
        z = self.instance_head(features)                  # Instance embeddings
        c = self.cluster_head(features)                   # Cluster logits
        
        return z, c
