"""
model.py
--------
ResNet-50 embedding network for Re-ID.

Design choice (and this is worth remembering for the interview): we
DO NOT train a classifier over identities. We strip the final
classification layer of a standard ImageNet-pretrained ResNet-50 and
replace it with a single linear layer that maps pooled features to a
128-dim embedding, L2-normalized so that cosine similarity == dot
product. The network is trained purely with batch-hard triplet loss,
which directly optimizes the thing we actually care about at test
time: whether same-identity images end up closer together in
embedding space than different-identity images.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights

from config import cfg


class EmbeddingNet(nn.Module):
    def __init__(self, embedding_dim: int = None, pretrained: bool = None):
        super().__init__()
        embedding_dim = embedding_dim or cfg.EMBEDDING_DIM
        pretrained = cfg.PRETRAINED if pretrained is None else pretrained

        weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        backbone = resnet50(weights=weights)

        # Re-ID convention: drop the stride in the last residual stage so
        # the final feature map has finer spatial resolution (16x8 instead
        # of 8x4 for a 256x128 input), which measurably helps retrieval.
        backbone.layer4[0].conv2.stride = (1, 1)
        backbone.layer4[0].downsample[0].stride = (1, 1)

        # Keep everything up to (and including) layer4; drop avgpool+fc.
        self.backbone = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool,
            backbone.layer1, backbone.layer2, backbone.layer3, backbone.layer4,
        )
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.embedding_head = nn.Linear(2048, embedding_dim, bias=False)
        self.bn_neck = nn.BatchNorm1d(embedding_dim)
        self.bn_neck.bias.requires_grad_(False)  # BNNeck trick (no bias before norm)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat_map = self.backbone(x)                      # [B, 2048, H, W]
        pooled = self.global_pool(feat_map).flatten(1)    # [B, 2048]
        embedding = self.embedding_head(pooled)           # [B, D]
        embedding = self.bn_neck(embedding)
        embedding = F.normalize(embedding, p=2, dim=1)     # unit-norm -> cosine sim
        return embedding


def build_model() -> EmbeddingNet:
    model = EmbeddingNet()
    return model.to(cfg.DEVICE)
