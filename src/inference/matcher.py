"""
Product similarity matching using FAISS vector search.
"""

import numpy as np
import torch
from PIL import Image

from src.models.clip_embedder import ProductEmbedder


class ProductMatcher:
    """Find similar products using learned embeddings and FAISS."""

    def __init__(self, model_path, index_path, metadata_path, device="cuda"):
        import faiss
        import sqlite3

        self.device = device
        self.embedder = ProductEmbedder()
        self.embedder.load_state_dict(torch.load(model_path, map_location=device))
        self.embedder.to(device)
        self.embedder.eval()

        self.index = faiss.read_index(index_path)
        self.db = sqlite3.connect(metadata_path)

    def find_similar(self, image, top_k=10):
        """Find top-K similar products for a query image."""
        embedding = self.embedder.embed_image(image).numpy().reshape(1, -1)

        distances, indices = self.index.search(embedding, top_k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue
            metadata = self._get_metadata(int(idx))
            results.append({
                "index": int(idx),
                "similarity": float(dist),
                "metadata": metadata,
            })

        return results

    def _get_metadata(self, idx):
        """Look up product metadata by index."""
        cursor = self.db.execute(
            "SELECT image_path, entity_name, value, unit FROM products WHERE id = ?",
            (idx,)
        )
        row = cursor.fetchone()
        if row:
            return {
                "image_path": row[0],
                "entity_name": row[1],
                "value": row[2],
                "unit": row[3],
            }
        return {}
