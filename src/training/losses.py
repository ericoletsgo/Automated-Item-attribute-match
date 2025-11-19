"""
Additional loss functions for contrastive and metric learning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class InfoNCELoss(nn.Module):
    """Standard InfoNCE / NT-Xent loss for self-supervised contrastive learning."""

    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, query, key):
        """
        Args:
            query: (batch_size, embed_dim) L2-normalized
            key: (batch_size, embed_dim) L2-normalized (augmented view)
        """
        batch_size = query.shape[0]

        # positive pairs are (query[i], key[i])
        pos_sim = (query * key).sum(dim=1) / self.temperature

        # all pairwise similarities
        all_sim = torch.matmul(query, key.T) / self.temperature

        # NCE: log softmax over positives
        labels = torch.arange(batch_size, device=query.device)
        loss = F.cross_entropy(all_sim, labels)
        return loss


class TripletLoss(nn.Module):
    """Triplet loss with hard negative mining."""

    def __init__(self, margin=0.3):
        super().__init__()
        self.margin = margin

    def forward(self, embeddings, labels):
        """Mine hard triplets within the batch."""
        dist_matrix = torch.cdist(embeddings, embeddings, p=2)

        labels = labels.unsqueeze(0)
        pos_mask = (labels == labels.T).float()
        neg_mask = (labels != labels.T).float()
        pos_mask.fill_diagonal_(0)

        # hardest positive: farthest positive for each anchor
        hard_pos = (dist_matrix * pos_mask).max(dim=1).values

        # hardest negative: closest negative for each anchor
        neg_dist = dist_matrix + (1 - neg_mask) * 1e6
        hard_neg = neg_dist.min(dim=1).values

        loss = F.relu(hard_pos - hard_neg + self.margin).mean()
        return loss
