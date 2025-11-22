"""
Build the FAISS index from all product embeddings.
Run after contrastive model training is complete.
"""

from pathlib import Path
from src.vectordb.indexer import build_index


def main():
    Path("data").mkdir(exist_ok=True)

    build_index(
        split_path="data/splits/train.json",
        image_dir="data/raw/images",
        model_path="checkpoints/contrastive/best_model.pt",
        output_index="data/faiss_index",
        output_db="data/metadata.db",
        batch_size=64,
        device="cuda",
    )


if __name__ == "__main__":
    main()
