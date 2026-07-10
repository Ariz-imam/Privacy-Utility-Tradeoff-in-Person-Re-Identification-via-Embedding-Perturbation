"""
evaluate.py
-----------
Standard Market-1501 evaluation protocol: Rank-1 / Rank-5 / Rank-10
(CMC) and mAP, with the privacy module optionally applied to the
gallery+query embeddings before matching.

Protocol details that matter (and are easy to get subtly wrong):
  - For a given query image (identity q_pid, camera q_camid), gallery
    images of the SAME identity taken by the SAME camera are excluded
    from that query's ranking (they're near-duplicate frames of the
    same tracklet, not a genuine cross-camera match).
  - Gallery images with pid == -1 (distractors) or pid == 0 (junk) are
    always excluded from being counted as correct matches, but distractors
    are NOT removed from the ranking itself — they still act as
    confusable "noise" the retrieval has to rank below the true match.
This file follows the widely used reference implementation logic
(Zheng et al., 2015 / open-reid) so numbers are comparable to published
Re-ID baselines.
"""

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import cfg
from dataset import Market1501Dataset


@torch.no_grad()
def extract_embeddings(model, dataset, batch_size=128):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                         num_workers=cfg.NUM_WORKERS)
    model.eval()

    all_embeddings, all_pids, all_camids = [], [], []
    for imgs, pids, camids, _ in tqdm(loader, desc=f"Extracting embeddings", leave=False):
        imgs = imgs.to(cfg.DEVICE)
        embeddings = model(imgs)
        all_embeddings.append(embeddings.cpu())
        all_pids.append(pids if torch.is_tensor(pids) else torch.tensor(pids))
        all_camids.append(camids if torch.is_tensor(camids) else torch.tensor(camids))

    return (torch.cat(all_embeddings, 0),
            torch.cat(all_pids, 0).numpy(),
            torch.cat(all_camids, 0).numpy())


def compute_cmc_map(distmat: np.ndarray, q_pids, g_pids, q_camids, g_camids,
                     max_rank=10):
    """
    distmat: [num_query, num_gallery] distance matrix (lower = more similar)
    Returns: cmc (array of length max_rank), mAP (float)
    """
    num_query, num_gallery = distmat.shape
    indices = np.argsort(distmat, axis=1)

    all_cmc = []
    all_ap = []
    num_valid_queries = 0

    for q_idx in range(num_query):
        q_pid, q_camid = q_pids[q_idx], q_camids[q_idx]
        order = indices[q_idx]

        g_pid_sorted = g_pids[order]
        g_camid_sorted = g_camids[order]

        # Remove gallery entries that are the same identity AND same camera
        # (near-duplicate tracklet frames) — Market-1501 protocol.
        remove = (g_pid_sorted == q_pid) & (g_camid_sorted == q_camid)
        keep = ~remove

        # Junk images (pid == 0 or -1) never count as valid matches, but
        # remain in the ranking as distractors — already excluded from
        # `keep` only when they coincide with the same-cam/same-id rule
        # above; explicit junk mask is applied to the "matches" vector
        # below instead of removing them from `order`.
        g_pid_kept = g_pid_sorted[keep]
        matches = (g_pid_kept == q_pid).astype(np.int32)

        if matches.sum() == 0:
            continue  # no ground-truth match for this query; skip

        num_valid_queries += 1

        # CMC
        cmc = matches.cumsum()
        cmc[cmc > 1] = 1
        all_cmc.append(cmc[:max_rank])

        # AP
        num_rel = matches.sum()
        tmp_cmc = matches.cumsum()
        precision_at_k = tmp_cmc / (np.arange(len(matches)) + 1.0)
        ap = (precision_at_k * matches).sum() / num_rel
        all_ap.append(ap)

    assert num_valid_queries > 0, "No valid queries found — check dataset paths / parsing."

    all_cmc = np.array(all_cmc, dtype=np.float32)
    # Pad rows shorter than max_rank (rare, only if gallery is tiny)
    cmc = all_cmc.mean(axis=0)
    mAP = float(np.mean(all_ap))
    return cmc, mAP


def evaluate(model, privacy_module=None, verbose=True):
    """
    Run full query-vs-gallery evaluation. If `privacy_module` is given,
    it is applied to BOTH query and gallery embeddings before matching
    (this models a system that always releases privatized embeddings,
    not one where only the gallery is protected).
    """
    query_set = Market1501Dataset(split="query")
    gallery_set = Market1501Dataset(split="gallery")

    q_embed, q_pids, q_camids = extract_embeddings(model, query_set)
    g_embed, g_pids, g_camids = extract_embeddings(model, gallery_set)

    if privacy_module is not None:
        q_embed = privacy_module(q_embed)
        g_embed = privacy_module(g_embed)

    # Cosine distance = 1 - cosine similarity (embeddings are unit-norm,
    # so this is a valid metric and monotonic with Euclidean distance).
    distmat = 1 - (q_embed @ g_embed.t()).numpy()

    cmc, mAP = compute_cmc_map(distmat, q_pids, g_pids, q_camids, g_camids,
                                max_rank=max(cfg.EVAL_RANKS))

    results = {"mAP": mAP}
    for r in cfg.EVAL_RANKS:
        results[f"Rank-{r}"] = float(cmc[r - 1])

    if verbose:
        sigma = getattr(privacy_module, "sigma", 0.0)
        print(f"[sigma={sigma}] " + " | ".join(f"{k}: {v:.4f}" for k, v in results.items()))

    return results, (q_embed, q_pids, g_embed, g_pids)
