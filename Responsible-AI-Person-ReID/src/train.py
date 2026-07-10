"""
train.py
--------
Training loop: fine-tune ResNet-50 with batch-hard triplet loss on
Market-1501, using PK sampling. Run directly (`python train.py`) or
import `train()` from a notebook.
"""

import os
import time

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

from config import cfg
from dataset import Market1501Dataset
from sampler import PKSampler
from model import build_model
from losses import BatchHardTripletLoss
from utils import set_seed, AverageMeter


def get_warmup_lr(epoch: int, base_lr: float, warmup_epochs: int) -> float:
    if epoch >= warmup_epochs:
        return base_lr
    # Linear warmup from 10% -> 100% of base_lr over `warmup_epochs`
    return base_lr * (0.1 + 0.9 * epoch / max(warmup_epochs, 1))


def train(num_epochs: int = None, resume_from: str = None):
    set_seed(cfg.SEED)

    train_set = Market1501Dataset(split="train")
    sampler = PKSampler(train_set)
    loader = DataLoader(train_set, batch_size=cfg.BATCH_SIZE, sampler=sampler,
                         num_workers=cfg.NUM_WORKERS, drop_last=True)

    print(f"Training identities: {train_set.num_ids} | "
          f"Training images: {len(train_set)} | "
          f"Batches/epoch: {len(loader)}")

    model = build_model()
    if resume_from and os.path.exists(resume_from):
        model.load_state_dict(torch.load(resume_from, map_location=cfg.DEVICE))
        print(f"Resumed weights from {resume_from}")

    criterion = BatchHardTripletLoss()
    optimizer = Adam(model.parameters(), lr=cfg.BASE_LR, weight_decay=cfg.WEIGHT_DECAY)
    scheduler = StepLR(optimizer, step_size=cfg.LR_STEP_SIZE, gamma=cfg.LR_GAMMA)

    num_epochs = num_epochs or cfg.NUM_EPOCHS
    history = []

    for epoch in range(1, num_epochs + 1):
        model.train()
        loss_meter = AverageMeter()
        active_meter = AverageMeter()

        # Manual warmup for the first few epochs, then hand control to StepLR
        if epoch <= cfg.WARMUP_EPOCHS:
            lr = get_warmup_lr(epoch - 1, cfg.BASE_LR, cfg.WARMUP_EPOCHS)
            for g in optimizer.param_groups:
                g["lr"] = lr

        t0 = time.time()
        for it, (imgs, labels, _, _) in enumerate(loader):
            imgs = imgs.to(cfg.DEVICE)
            labels = labels.to(cfg.DEVICE)

            embeddings = model(imgs)
            loss, active_frac = criterion(embeddings, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_meter.update(loss.item(), imgs.size(0))
            active_meter.update(active_frac, imgs.size(0))

            if (it + 1) % cfg.LOG_INTERVAL == 0:
                print(f"Epoch [{epoch}/{num_epochs}] Iter [{it+1}/{len(loader)}] "
                      f"Loss: {loss_meter.avg:.4f} | Active triplet frac: {active_meter.avg:.3f} "
                      f"| LR: {optimizer.param_groups[0]['lr']:.6f}")

        if epoch > cfg.WARMUP_EPOCHS:
            scheduler.step()

        epoch_time = time.time() - t0
        print(f"== Epoch {epoch} done in {epoch_time:.1f}s | Avg loss: {loss_meter.avg:.4f} ==")
        history.append({"epoch": epoch, "loss": loss_meter.avg, "active_frac": active_meter.avg})

        if epoch % cfg.SAVE_EVERY == 0 or epoch == num_epochs:
            ckpt_path = os.path.join(cfg.CHECKPOINT_DIR, f"embedding_net_epoch{epoch}.pth")
            torch.save(model.state_dict(), ckpt_path)
            print(f"Saved checkpoint: {ckpt_path}")

    final_path = os.path.join(cfg.CHECKPOINT_DIR, "embedding_net_final.pth")
    torch.save(model.state_dict(), final_path)
    print(f"Training complete. Final weights: {final_path}")

    return model, history


if __name__ == "__main__":
    train()
