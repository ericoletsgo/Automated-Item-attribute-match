"""
PyTorch datasets for Florence-2 fine-tuning and contrastive training.
"""

import json
from pathlib import Path

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from PIL import Image


class Florence2Dataset(Dataset):
    """Dataset for Florence-2 attribute extraction fine-tuning."""

    def __init__(self, split_path, image_dir, processor):
        with open(split_path) as f:
            self.records = json.load(f)
        self.image_dir = Path(image_dir)
        self.processor = processor

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record = self.records[idx]

        image_path = self.image_dir / self._get_image_filename(record["image_link"])
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception:
            image = Image.new("RGB", (768, 768))

        prompt = "<OCR>"
        target = record.get("target", f"{record.get('value', '')} {record.get('unit', '')}")

        inputs = self.processor(
            text=prompt,
            images=image,
            return_tensors="pt",
        )

        labels = self.processor.tokenizer(
            target,
            return_tensors="pt",
            max_length=128,
            padding="max_length",
            truncation=True,
        )

        return {
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "pixel_values": inputs["pixel_values"].squeeze(0),
            "labels": labels["input_ids"].squeeze(0),
        }

    def _get_image_filename(self, url):
        import hashlib
        url_hash = hashlib.md5(url.encode()).hexdigest()
        ext = url.split(".")[-1].split("?")[0]
        if ext not in ("jpg", "jpeg", "png", "webp"):
            ext = "jpg"
        return f"{url_hash}.{ext}"


def florence2_collate_fn(batch):
    """Custom collate that pads input_ids and attention_mask to same length."""
    batch = [b for b in batch if b is not None]
    if not batch:
        return None

    # pad input_ids and attention_mask to max length in batch
    input_ids = pad_sequence([b["input_ids"] for b in batch], batch_first=True, padding_value=1)
    attention_mask = pad_sequence([b["attention_mask"] for b in batch], batch_first=True, padding_value=0)

    # pixel_values and labels are already fixed size
    pixel_values = torch.stack([b["pixel_values"] for b in batch])
    labels = torch.stack([b["labels"] for b in batch])

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "pixel_values": pixel_values,
        "labels": labels,
    }


class ContrastiveDataset(Dataset):
    """Dataset for contrastive learning with product images."""

    def __init__(self, split_path, image_dir, transform=None):
        with open(split_path) as f:
            self.records = json.load(f)
        self.image_dir = Path(image_dir)
        self.transform = transform

        self.groups = {}
        for i, r in enumerate(self.records):
            key = r["entity_name"]
            if key not in self.groups:
                self.groups[key] = []
            self.groups[key].append(i)

        entity_names = sorted(set(r["entity_name"] for r in self.records))
        self.label_map = {name: i for i, name in enumerate(entity_names)}

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record = self.records[idx]

        image_path = self.image_dir / self._get_image_filename(record["image_link"])
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception:
            image = Image.new("RGB", (224, 224))

        if self.transform:
            image = self.transform(image)

        label = self.label_map[record["entity_name"]]
        return image, label

    def _get_image_filename(self, url):
        import hashlib
        url_hash = hashlib.md5(url.encode()).hexdigest()
        ext = url.split(".")[-1].split("?")[0]
        if ext not in ("jpg", "jpeg", "png", "webp"):
            ext = "jpg"
        return f"{url_hash}.{ext}"
