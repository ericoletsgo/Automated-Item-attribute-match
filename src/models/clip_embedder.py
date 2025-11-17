"""
SigLIP-based image embedder for product similarity.
"""

import torch
import torch.nn as nn
import open_clip
from PIL import Image


class ProductEmbedder(nn.Module):
    """Embeds product images into a learned similarity space."""

    def __init__(self, backbone="ViT-B-16-SigLIP", pretrained="webli", projection_dims=None):
        super().__init__()

        if projection_dims is None:
            projection_dims = [768, 512, 256]

        self.backbone, _, self.preprocess = open_clip.create_model_and_transforms(
            backbone, pretrained=pretrained
        )

        # freeze backbone initially
        for param in self.backbone.parameters():
            param.requires_grad = False

        # projection head
        layers = []
        for i in range(len(projection_dims) - 1):
            layers.append(nn.Linear(projection_dims[i], projection_dims[i + 1]))
            if i < len(projection_dims) - 2:
                layers.append(nn.BatchNorm1d(projection_dims[i + 1]))
                layers.append(nn.ReLU())
        self.projector = nn.Sequential(*layers)

        self.embed_dim = projection_dims[-1]

    def forward(self, images):
        """Get L2-normalized embeddings for a batch of images."""
        with torch.set_grad_enabled(self.backbone.training):
            features = self.backbone.encode_image(images)

        projected = self.projector(features.float())
        return nn.functional.normalize(projected, dim=1)

    def unfreeze_backbone(self, num_layers=2):
        """Unfreeze the last N transformer blocks for fine-tuning."""
        visual = self.backbone.visual
        blocks = list(visual.transformer.resblocks) if hasattr(visual, 'transformer') else []

        if not blocks:
            # try alternate attribute names
            if hasattr(visual, 'trunk'):
                blocks = list(visual.trunk.blocks)

        for block in blocks[-num_layers:]:
            for param in block.parameters():
                param.requires_grad = True

    @torch.no_grad()
    def embed_image(self, image):
        """Embed a single PIL image."""
        self.eval()
        if isinstance(image, Image.Image):
            image = self.preprocess(image).unsqueeze(0)
        if next(self.parameters()).is_cuda:
            image = image.cuda()
        return self(image).cpu().squeeze(0)
