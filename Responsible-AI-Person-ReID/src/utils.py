"""
utils.py
--------
Small, boring, reusable helpers. Nothing here is specific to Re-ID —
seeding, timing, and running averages that any training script needs.
"""

import random
import time
import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Make runs reproducible across numpy / torch / python's random."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class AverageMeter:
    """Tracks a running average of a scalar (loss, accuracy, etc.)."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / max(self.count, 1)


class Timer:
    """Context manager to time a block of code."""

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.elapsed = time.time() - self.start


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
