"""
Build and manage FAISS index for product similarity search.
"""

import json
import sqlite3
from pathlib import Path

import numpy as np
import torch
import faiss
from PIL import Image
from tqdm import tqdm

from src.models.clip_embedder import ProductEmbedder


def build_index(
    split_path="data/splits/train.json",
    image_dir="data/raw/images",
    model_path="checkpoints/contrastive/best_model.pt",
    output_index="data/faiss_index",
    output_db="data/metadata.db",
    batch_size=64,
    device="cuda",
):
    """Build FAISS index from all product images."""

    # load model
    embedder = ProductEmbedder()
    embedder.load_state_dict(torch.load(model_path, map_location=device))
    embedder.to(device)
    embedder.eval()

    # load data
    with open(split_path) as f:
        records = json.load(f)

    # setup metadata db
    db = sqlite3.connect(output_db)
    db.execute("""
        CREATE TABLE IF NOT EXISTS products (
            id INTEGER PRIMARY KEY,
            image_path TEXT,
            entity_name TEXT,
            value REAL,
            unit TEXT
        )
    """)

    # embed all images
    all_embeddings = []
    image_dir = Path(image_dir)

    for i in tqdm(range(0, len(records), batch_size), desc="Building index"):
        batch_records = records[i:i + batch_size]
        images = []

        for record in batch_records:
            img_path = image_dir / _get_image_filename(record["image_link"])
            try:
                img = Image.open(img_path).convert("RGB")
                img = embedder.preprocess(img)
            except Exception:
                img = torch.zeros(3, 224, 224)
            images.append(img)

        images_tensor = torch.stack(images).to(device)
        with torch.no_grad():
            embeddings = embedder(images_tensor).cpu().numpy()
        all_embeddings.append(embeddings)

        # store metadata
        for j, record in enumerate(batch_records):
            db.execute(
                "INSERT OR REPLACE INTO products (id, image_path, entity_name, value, unit) VALUES (?, ?, ?, ?, ?)",
                (i + j, record["image_link"], record["entity_name"], record["value"], record["unit"]),
            )

    db.commit()
    db.close()

    # build FAISS index
    all_embeddings = np.vstack(all_embeddings).astype(np.float32)
    dim = all_embeddings.shape[1]

    # IVF index for fast approximate search
    quantizer = faiss.IndexFlatIP(dim)
    index = faiss.IndexIVFFlat(quantizer, dim, min(256, len(all_embeddings) // 10), faiss.METRIC_INNER_PRODUCT)
    index.train(all_embeddings)
    index.add(all_embeddings)

    faiss.write_index(index, output_index)
    print(f"Built index with {index.ntotal} vectors, dim={dim}")


def _get_image_filename(url):
    import hashlib
    url_hash = hashlib.md5(url.encode()).hexdigest()
    ext = url.split(".")[-1].split("?")[0]
    if ext not in ("jpg", "jpeg", "png", "webp"):
        ext = "jpg"
    return f"{url_hash}.{ext}"


if __name__ == "__main__":
    build_index()
