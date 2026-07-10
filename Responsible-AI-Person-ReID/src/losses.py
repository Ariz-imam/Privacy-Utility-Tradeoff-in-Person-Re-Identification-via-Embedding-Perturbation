"""
losses.py
---------
Batch-hard triplet loss (Hermans et al., "In Defense of the Triplet
Loss for Person Re-Identification", 2017).

For every anchor in the batch we don't use all possible triplets
(there are far too many, and most are "easy" and give near-zero
gradient). Instead, for each anchor we pick:
    - the HARDEST positive: same identity, largest distance
    - the HARDEST negative: different identity, smallest distance
and train on that one triplet. This is the standard, well-understood
recipe most Re-ID papers use as a baseline — easy to explain in an
interview, and it works well when paired with PK sampling.
"""

import torch
import torch.nn as nn

from config import cfg


def pairwise_euclidean_distance(embeddings: torch.Tensor) -> torch.Tensor:
    """
    Compute a [B, B] matrix of pairwise Euclidean distances.
    Embeddings are L2-normalized, so this is monotonic with cosine
    distance — using Euclidean here just keeps the margin interpretable.
    """
    dot_product = embeddings @ embeddings.t()
    squared_norm = torch.diagonal(dot_product)
    distances = squared_norm.unsqueeze(0) - 2 * dot_product + squared_norm.unsqueeze(1)
    distances = torch.clamp(distances, min=0.0)
    # sqrt with a tiny epsilon to avoid NaN gradients at distance == 0
    return torch.sqrt(distances + 1e-12)


class BatchHardTripletLoss(nn.Module):
    def __init__(self, margin: float = None):
        super().__init__()
        self.margin = margin or cfg.TRIPLET_MARGIN
        self.ranking_loss = nn.MarginRankingLoss(margin=self.margin)

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor):
        distances = pairwise_euclidean_distance(embeddings)
        labels = labels.unsqueeze(1)
        same_identity_mask = (labels == labels.t())
        diff_identity_mask = ~same_identity_mask

        # Hardest positive: max distance among same-identity pairs
        # (mask out self-comparisons which are always distance 0 by
        # setting them to -inf before the max).
        pos_dist = distances.clone()
        pos_dist[~same_identity_mask] = float("-inf")
        pos_dist.fill_diagonal_(float("-inf"))
        hardest_positive, _ = pos_dist.max(dim=1)

        # Hardest negative: min distance among different-identity pairs
        neg_dist = distances.clone()
        neg_dist[~diff_identity_mask] = float("inf")
        hardest_negative, _ = neg_dist.min(dim=1)

        y = torch.ones_like(hardest_positive)
        loss = self.ranking_loss(hardest_negative, hardest_positive, y)

        # Useful diagnostic: fraction of triplets where the margin is
        # already satisfied (i.e. "easy" triplets). Logged during training.
        with torch.no_grad():
            active_fraction = (hardest_negative - hardest_positive < self.margin).float().mean()

        return loss, active_fraction.item()
