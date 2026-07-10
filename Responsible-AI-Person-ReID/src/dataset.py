"""
dataset.py
----------
Market-1501 loading utilities.

Market-1501 filenames follow the pattern:

    0002_c1s1_000451_03.jpg
    ^^^^ ^^ ^^^^^^ ^^
    |    |  |      |
    |    |  |      +-- image index for that tracklet
    |    |  +--------- frame number
    |    +------------ camera id (c1 ... c6)
    +----------------- person identity (0000 = junk, -1 = distractor)

For the training split we keep every identity except junk (0000).
For query/gallery we keep everything (including junk/distractor,
which the evaluation protocol needs to correctly ignore).
"""

import os
import re
from typing import List, Tuple

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from config import cfg

FILENAME_PATTERN = re.compile(r"([-\d]+)_c(\d)")


def parse_filename(filename: str) -> Tuple[int, int]:
    """Return (person_id, camera_id) parsed from a Market-1501 filename."""
    match = FILENAME_PATTERN.search(filename)
    if match is None:
        raise ValueError(f"Filename does not match Market-1501 pattern: {filename}")
    person_id, camera_id = int(match.group(1)), int(match.group(2))
    return person_id, camera_id


def list_images(directory: str) -> List[str]:
    return sorted(
        f for f in os.listdir(directory)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    )


def build_transform(is_train: bool) -> transforms.Compose:
    """
    Standard Re-ID augmentation. Random erasing is a well-known trick
    in Re-ID literature (simulates occlusion) and gives a real accuracy
    bump for very little complexity, so it's included in training only.
    """
    if is_train:
        return transforms.Compose([
            transforms.Resize((cfg.IMAGE_HEIGHT, cfg.IMAGE_WIDTH)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.Pad(10),
            transforms.RandomCrop((cfg.IMAGE_HEIGHT, cfg.IMAGE_WIDTH)),
            transforms.ToTensor(),
            transforms.Normalize(cfg.IMAGENET_MEAN, cfg.IMAGENET_STD),
            transforms.RandomErasing(p=0.5, scale=(0.02, 0.2)),
        ])
    return transforms.Compose([
        transforms.Resize((cfg.IMAGE_HEIGHT, cfg.IMAGE_WIDTH)),
        transforms.ToTensor(),
        transforms.Normalize(cfg.IMAGENET_MEAN, cfg.IMAGENET_STD),
    ])


class Market1501Dataset(Dataset):
    """
    Generic Market-1501 split reader.

    split: one of "train", "query", "gallery" — selects the directory
    and whether junk/distractor identities (id == 0 or id == -1) are
    dropped (only dropped for "train").
    """

    def __init__(self, split: str, transform=None):
        assert split in {"train", "query", "gallery"}
        self.split = split
        directory = {
            "train": cfg.TRAIN_DIR,
            "query": cfg.QUERY_DIR,
            "gallery": cfg.GALLERY_DIR,
        }[split]
        self.directory = directory
        self.transform = transform or build_transform(is_train=(split == "train"))

        filenames = list_images(directory)
        samples = []
        for fname in filenames:
            pid, camid = parse_filename(fname)
            if split == "train" and pid in (0, -1):
                continue  # drop junk / distractor identities from training
            samples.append((fname, pid, camid))
        self.samples = samples

        # Re-map raw person IDs to a contiguous [0, num_train_ids) range,
        # required for any classification-style auxiliary loss and for
        # clean PK-sampling bookkeeping.
        unique_ids = sorted({pid for _, pid, _ in samples})
        self.pid_to_label = {pid: i for i, pid in enumerate(unique_ids)}
        self.num_ids = len(unique_ids)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        fname, pid, camid = self.samples[index]
        img = Image.open(os.path.join(self.directory, fname)).convert("RGB")
        img = self.transform(img)
        label = self.pid_to_label[pid] if self.split == "train" else pid
        return img, label, camid, fname
