"""
FAISS index for similar products.
"""

import numpy as np
import faiss


class VectorSearch:
    """FAISS index for product search."""

    def __init__(self, index_path, nprobe=16):
        self.index = faiss.read_index(index_path)
        if hasattr(self.index, "nprobe"):
            self.index.nprobe = nprobe

    def search(self, query_embedding, top_k=10):
        """Search for nearest neighbors.

        Args:
            query_embedding: numpy array of shape (dim,) or (1, dim)
            top_k: number of results to return

        Returns:
            list of (index, score) tuples
        """
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        query_embedding = query_embedding.astype(np.float32)
        distances, indices = self.index.search(query_embedding, top_k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx != -1:
                results.append((int(idx), float(dist)))

        return results

    @property
    def size(self):
        return self.index.ntotal
