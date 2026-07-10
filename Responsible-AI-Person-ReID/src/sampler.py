"""
sampler.py
----------
PK sampler: every batch contains P identities x K images per identity.

Batch-hard triplet loss needs at least 2 images per identity in a batch
(so a positive pair exists) and more than 1 identity (so a negative
exists). PK sampling guarantees both, every batch, by construction —
this is the standard sampling strategy used in Re-ID training and is
much simpler to reason about than mining triplets from a randomly
shuffled loader.
"""

import copy
import random
from collections import defaultdict

from torch.utils.data.sampler import Sampler

from config import cfg


class PKSampler(Sampler):
    def __init__(self, dataset, num_ids_per_batch=None, num_images_per_id=None):
        self.num_ids_per_batch = num_ids_per_batch or cfg.NUM_IDS_PER_BATCH
        self.num_images_per_id = num_images_per_id or cfg.NUM_IMAGES_PER_ID

        # Map: label -> list of dataset indices belonging to that identity
        self.label_to_indices = defaultdict(list)
        for idx, (_, pid, _) in enumerate(dataset.samples):
            label = dataset.pid_to_label[pid]
            self.label_to_indices[label].append(idx)

        # Drop identities with fewer than 2 images: no positive pair possible.
        self.labels = [
            l for l, idxs in self.label_to_indices.items() if len(idxs) >= 2
        ]

        self.length = len(self.labels) // self.num_ids_per_batch \
            * self.num_ids_per_batch * self.num_images_per_id

    def __len__(self):
        return self.length

    def __iter__(self):
        batch_indices = []
        label_pool = copy.deepcopy(self.labels)
        random.shuffle(label_pool)

        # Pre-shuffle each identity's image list so repeated epochs see
        # different K-subsets when an identity has more than K images.
        indices_copy = {l: copy.deepcopy(v) for l, v in self.label_to_indices.items()}
        for v in indices_copy.values():
            random.shuffle(v)

        while len(label_pool) >= self.num_ids_per_batch:
            chosen_labels = [label_pool.pop() for _ in range(self.num_ids_per_batch)]
            for label in chosen_labels:
                idxs = indices_copy[label]
                if len(idxs) >= self.num_images_per_id:
                    picked = idxs[: self.num_images_per_id]
                else:
                    # Sample with replacement if the identity has < K images
                    picked = [random.choice(idxs) for _ in range(self.num_images_per_id)]
                batch_indices.extend(picked)

        return iter(batch_indices)
