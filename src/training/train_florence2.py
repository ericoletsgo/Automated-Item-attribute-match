"""
Fine-tune Florence-2 with LoRA for product attribute extraction.
"""

import yaml
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
import wandb

from src.models.florence2_extractor import Florence2Extractor
from src.data.dataset import Florence2Dataset


def train(config_path="configs/florence2_finetune.yaml"):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    wandb.init(project="product-matcher", name="florence2-finetune", config=config)

    model, processor = Florence2Extractor.setup_for_training(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    train_dataset = Florence2Dataset(
        split_path=config["data"]["train_split"],
        image_dir="data/raw/images",
        processor=processor,
    )
    val_dataset = Florence2Dataset(
        split_path=config["data"]["val_split"],
        image_dir="data/raw/images",
        processor=processor,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["training"]["batch_size"],
        num_workers=4,
    )

    optimizer = AdamW(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=0.01,
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config["training"]["warmup_steps"],
        num_training_steps=config["training"]["max_steps"],
    )

    scaler = torch.amp.GradScaler("cuda")
    global_step = 0
    best_val_loss = float("inf")

    model.train()
    while global_step < config["training"]["max_steps"]:
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}

            with torch.amp.autocast("cuda"):
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    pixel_values=batch["pixel_values"],
                    labels=batch["labels"],
                )
                loss = outputs.loss / config["training"]["gradient_accumulation_steps"]

            scaler.scale(loss).backward()

            if (global_step + 1) % config["training"]["gradient_accumulation_steps"] == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()

            if global_step % config["training"]["logging_steps"] == 0:
                wandb.log({"train/loss": loss.item(), "lr": scheduler.get_last_lr()[0]}, step=global_step)

            if global_step % config["training"]["eval_steps"] == 0 and global_step > 0:
                val_loss = evaluate(model, val_loader, device)
                wandb.log({"val/loss": val_loss}, step=global_step)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    model.save_pretrained("checkpoints/florence2-lora")
                    print(f"Step {global_step}: saved best model (val_loss={val_loss:.4f})")

                model.train()

            global_step += 1
            if global_step >= config["training"]["max_steps"]:
                break

    wandb.finish()
    print("Training complete.")


@torch.no_grad()
def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0
    count = 0

    for batch in dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.amp.autocast("cuda"):
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                pixel_values=batch["pixel_values"],
                labels=batch["labels"],
            )
        total_loss += outputs.loss.item()
        count += 1

    return total_loss / max(count, 1)


if __name__ == "__main__":
    train()
