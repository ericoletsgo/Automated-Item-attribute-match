"""
Fine-tune Florence-2 with LoRA for product attribute extraction.
"""

import os
import sys
import time
import yaml
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
import wandb

from src.models.florence2_extractor import Florence2Extractor
from src.data.dataset import Florence2Dataset, florence2_collate_fn


def log(msg):
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


CHECKPOINT_DIR = "checkpoints/florence2-lora"


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
        num_workers=0,
        pin_memory=True,
        collate_fn=florence2_collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["training"]["batch_size"],
        num_workers=0,
        collate_fn=florence2_collate_fn,
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
    best_val_loss = float("inf")

    global_step = 0
    resume_path = os.path.join(CHECKPOINT_DIR, "training_state.pt")
    if os.path.exists(resume_path):
        state = torch.load(resume_path, map_location=device)
        global_step = state["global_step"]
        best_val_loss = state["best_val_loss"]
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        log(f"Resumed from step {global_step}")

    accum_steps = config["training"]["gradient_accumulation_steps"]
    max_steps = config["training"]["max_steps"]

    model.train()
    optimizer.zero_grad()

    log(f"Starting training: step {global_step} -> {max_steps}, device={device}")
    log(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    log(f"Batch size: {config['training']['batch_size']}, Accum: {accum_steps}, Effective: {config['training']['batch_size'] * accum_steps}")

    start_step = global_step
    step_start = time.time()
    last_heartbeat = time.time()

    while global_step < max_steps:
        for batch in train_loader:
            if batch is None:
                continue

            batch = {k: v.to(device) for k, v in batch.items()}

            try:
                with torch.amp.autocast("cuda"):
                    outputs = model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        pixel_values=batch["pixel_values"],
                        labels=batch["labels"],
                    )
                    loss = outputs.loss / accum_steps

                scaler.scale(loss).backward()
            except Exception as e:
                log(f"Step {global_step}: skipping batch - {e}")
                optimizer.zero_grad()
                continue

            if (global_step + 1) % accum_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()

            if global_step % config["training"]["logging_steps"] == 0:
                actual_loss = loss.item() * accum_steps
                elapsed = time.time() - step_start
                steps_done = max(global_step - start_step, 1)
                sec_per_step = elapsed / steps_done
                remaining = (max_steps - global_step) * sec_per_step
                eta_h, eta_m = int(remaining // 3600), int((remaining % 3600) // 60)
                log(f"Step {global_step}/{max_steps}: loss={actual_loss:.4f} | {sec_per_step:.1f}s/step | ETA {eta_h}h{eta_m}m")
                wandb.log({"train/loss": actual_loss, "lr": scheduler.get_last_lr()[0]}, step=global_step)

            elif time.time() - last_heartbeat > 120:
                log(f"Step {global_step}/{max_steps}: still training...")
                last_heartbeat = time.time()

            if global_step % config["training"]["save_steps"] == 0 and global_step > 0:
                save_checkpoint(model, optimizer, scheduler, global_step, best_val_loss)

            if global_step % config["training"]["eval_steps"] == 0 and global_step > 0:
                log(f"Step {global_step}: starting eval (max 50 batches)...")
                eval_start = time.time()
                val_loss = evaluate(model, val_loader, device)
                eval_time = time.time() - eval_start
                wandb.log({"val/loss": val_loss}, step=global_step)
                log(f"Step {global_step}: val_loss={val_loss:.4f} (eval took {eval_time:.0f}s)")

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    model.save_pretrained(CHECKPOINT_DIR)
                    log(f"Step {global_step}: new best model saved")

                model.train()

            global_step += 1
            if global_step >= max_steps:
                break

    save_checkpoint(model, optimizer, scheduler, global_step, best_val_loss)
    model.save_pretrained(CHECKPOINT_DIR)
    wandb.finish()
    log("Training complete.")


def save_checkpoint(model, optimizer, scheduler, global_step, best_val_loss):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    torch.save({
        "global_step": global_step,
        "best_val_loss": best_val_loss,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
    }, os.path.join(CHECKPOINT_DIR, "training_state.pt"))
    log(f"Step {global_step}: checkpoint saved")


@torch.no_grad()
def evaluate(model, dataloader, device, max_batches=50):
    model.eval()
    total_loss = 0
    count = 0

    for batch in dataloader:
        if batch is None:
            continue
        if count >= max_batches:
            break
        batch = {k: v.to(device) for k, v in batch.items()}
        try:
            with torch.amp.autocast("cuda"):
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    pixel_values=batch["pixel_values"],
                    labels=batch["labels"],
                )
            total_loss += outputs.loss.item()
            count += 1
            if count % 10 == 0:
                log(f"  eval batch {count}/{max_batches}...")
        except Exception:
            continue

    return total_loss / max(count, 1)


if __name__ == "__main__":
    train()
