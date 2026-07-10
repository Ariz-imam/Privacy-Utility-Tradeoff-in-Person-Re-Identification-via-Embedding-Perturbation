"""
config.py
---------
Central configuration for the Privacy-Utility Tradeoff in Person
Re-Identification project.

Every path, hyperparameter and privacy setting used anywhere in the
pipeline is defined here so the rest of the code never hard-codes a
number. Change values here, not in train.py / evaluate.py.
"""

import os
import torch


class Config:
    # ------------------------------------------------------------------
    # Paths
    # ------------------------------------------------------------------
    # On Kaggle, the Market-1501 dataset (added via "Add Data") will be
    # mounted under /kaggle/input/<dataset-slug>/. Update DATASET_ROOT to
    # match the exact folder Kaggle mounts it at (print it once with
    # `os.listdir('/kaggle/input')` if unsure).
    if os.path.exists("/kaggle/input"):
        DATASET_ROOT = "/kaggle/input/market-1501/Market-1501-v15.09.15"
        OUTPUT_ROOT = "/kaggle/working"
    else:
        # Local / non-Kaggle fallback (edit as needed)
        DATASET_ROOT = "./data/Market-1501-v15.09.15"
        OUTPUT_ROOT = "."

    TRAIN_DIR = os.path.join(DATASET_ROOT, "bounding_box_train")
    QUERY_DIR = os.path.join(DATASET_ROOT, "query")
    GALLERY_DIR = os.path.join(DATASET_ROOT, "bounding_box_test")

    CHECKPOINT_DIR = os.path.join(OUTPUT_ROOT, "checkpoints")
    RESULTS_DIR = os.path.join(OUTPUT_ROOT, "results")
    FIGURES_DIR = os.path.join(OUTPUT_ROOT, "figures")

    # ------------------------------------------------------------------
    # Reproducibility / device
    # ------------------------------------------------------------------
    SEED = 42
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ------------------------------------------------------------------
    # Image preprocessing
    # ------------------------------------------------------------------
    IMAGE_HEIGHT = 256
    IMAGE_WIDTH = 128
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]

    # ------------------------------------------------------------------
    # PK-sampling (P identities x K images per batch) — standard for
    # batch-hard triplet mining in Re-ID.
    # ------------------------------------------------------------------
    NUM_IDS_PER_BATCH = 16          # P
    NUM_IMAGES_PER_ID = 4           # K
    BATCH_SIZE = NUM_IDS_PER_BATCH * NUM_IMAGES_PER_ID   # 64

    NUM_WORKERS = 4

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    EMBEDDING_DIM = 128
    BACKBONE = "resnet50"           # feature extractor
    PRETRAINED = True

    # ------------------------------------------------------------------
    # Optimization
    # ------------------------------------------------------------------
    NUM_EPOCHS = 60
    BASE_LR = 3.5e-4
    WEIGHT_DECAY = 5e-4
    LR_STEP_SIZE = 20               # StepLR: decay every N epochs
    LR_GAMMA = 0.1
    WARMUP_EPOCHS = 5

    TRIPLET_MARGIN = 0.3            # margin for batch-hard triplet loss

    LOG_INTERVAL = 20               # print every N iterations
    SAVE_EVERY = 10                 # checkpoint every N epochs

    # ------------------------------------------------------------------
    # Privacy module (Gaussian noise on the embedding)
    # ------------------------------------------------------------------
    # sigma = 0.0 reproduces the non-private baseline.
    NOISE_SIGMA_LEVELS = [0.0, 0.05, 0.10, 0.20, 0.30]

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    EVAL_RANKS = [1, 5, 10]


cfg = Config()

for d in [cfg.CHECKPOINT_DIR, cfg.RESULTS_DIR, cfg.FIGURES_DIR]:
    os.makedirs(d, exist_ok=True)
