"""
Two-stage contrastive training for product similarity embeddings.
Stage 1: train projection head only backbone frozen
Stage 2: unfreeze last N backbone layers, fine-tune end-to-end
"""

import yaml
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import wandb

from src.models.clip_embedder import ProductEmbedder
from src.models.contrastive_head import SupConLoss
from src.data.dataset import ContrastiveDataset
from src.data.preprocessing import get_train_transform, get_val_transform, AlbumentationsWrapper


def train(config_path="configs/contrastive.yaml"):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    wandb.init(project="product-matcher", name="contrastive-training", config=config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ProductEmbedder(
        projection_dims=config["model"]["projection_dims"],
    ).to(device)

    criterion = SupConLoss(temperature=config["training"]["temperature"])

    train_transform = AlbumentationsWrapper(get_train_transform(config["data"]["image_size"]))
    val_transform = AlbumentationsWrapper(get_val_transform(config["data"]["image_size"]))

    train_dataset = ContrastiveDataset(
        split_path="data/splits/train.json",
        image_dir="data/raw/images",
        transform=train_transform,
    )
    val_dataset = ContrastiveDataset(
        split_path="data/splits/val.json",
        image_dir="data/raw/images",
        transform=val_transform,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["training"]["batch_size"],
        num_workers=0,
        drop_last=True,
    )

    # stage 1: only projection head
    optimizer = AdamW(
        model.projector.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=config["training"]["stage1_epochs"])

    print("Stage 1: training projection head only")
    for epoch in range(config["training"]["stage1_epochs"]):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = validate(model, val_loader, criterion, device)
        scheduler.step()

        wandb.log({"stage1/train_loss": train_loss, "stage1/val_loss": val_loss, "epoch": epoch})
        print(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")

    # stage 2: unfreeze backbone layers
    print(f"\nStage 2: unfreezing last {config['model']['unfreeze_layers']} backbone layers")
    model.unfreeze_backbone(config["model"]["unfreeze_layers"])

    all_params = [
        {"params": model.projector.parameters(), "lr": config["training"]["learning_rate"]},
        {"params": [p for p in model.backbone.parameters() if p.requires_grad],
         "lr": config["training"]["finetune_learning_rate"]},
    ]
    optimizer = AdamW(all_params, weight_decay=config["training"]["weight_decay"])
    remaining_epochs = config["training"]["epochs"] - config["training"]["stage1_epochs"]
    scheduler = CosineAnnealingLR(optimizer, T_max=remaining_epochs)

    best_val_loss = float("inf")
    for epoch in range(remaining_epochs):
        actual_epoch = epoch + config["training"]["stage1_epochs"]
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = validate(model, val_loader, criterion, device)
        scheduler.step()

        wandb.log({"stage2/train_loss": train_loss, "stage2/val_loss": val_loss, "epoch": actual_epoch})
        print(f"Epoch {actual_epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "checkpoints/contrastive/best_model.pt")

    wandb.finish()
    print("Training complete.")


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0

    for images, labels in tqdm(loader, desc="Training"):
        images = images.to(device)
        labels = labels.to(device)

        embeddings = model(images)
        loss = criterion(embeddings, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        embeddings = model(images)
        loss = criterion(embeddings, labels)
        total_loss += loss.item()

    return total_loss / len(loader)


if __name__ == "__main__":
    train()
