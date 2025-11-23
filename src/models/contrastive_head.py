"""
Supervised contrastive loss for product similarity learning.
"""

import torch
import torch.nn as nn


class SupConLoss(nn.Module):
    """Supervised Contrastive Loss.

    Products with the same label are treated as positives.
    Uses in-batch negatives with temperature scaling.
    """

    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        device = features.device
        batch_size = features.shape[0]

        if batch_size < 2:
            return torch.tensor(0.0, device=device, requires_grad=True)

        # similarity matrix
        sim = torch.matmul(features, features.T) / self.temperature

        # mask: 1 where labels match except self
        labels = labels.view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)
        mask.fill_diagonal_(0)

        # skip if no positive pairs in this batch
        if mask.sum() == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)

        # for numerical stability
        sim_max, _ = sim.max(dim=1, keepdim=True)
        sim = sim - sim_max.detach()

        # log-sum-exp over all negatives except self
        self_mask = torch.ones_like(sim).fill_diagonal_(0)
        exp_sim = torch.exp(sim) * self_mask
        log_sum_exp = torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-6)

        # mean of log-prob over positives
        log_prob = sim - log_sum_exp
        num_positives = mask.sum(dim=1)
        positive_log_prob = (mask * log_prob).sum(dim=1) / (num_positives + 1e-6)

        # only average over samples that have at least one positive
        valid = num_positives > 0
        if valid.sum() == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)

        loss = -positive_log_prob[valid].mean()
        return loss
