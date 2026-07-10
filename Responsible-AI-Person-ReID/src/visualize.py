"""
visualize.py
------------
t-SNE visualization of the embedding space, before vs after the
privacy module is applied. This is the single figure the project
report leans on to make the privacy-utility tradeoff visually
intuitive: identity clusters that are tight and well-separated at
sigma=0 should visibly blur together as sigma increases.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from config import cfg


def plot_tsne_comparison(embeddings_before: np.ndarray, embeddings_after: np.ndarray,
                          pids: np.ndarray, sigma: float, num_ids_to_plot: int = 15,
                          save_name: str = "tsne_comparison.png"):
    """
    embeddings_before / embeddings_after: [N, D] arrays (same N, same
    identity order) — typically gallery embeddings at sigma=0 vs a
    chosen sigma > 0.
    pids: [N] identity labels aligned with the embedding rows.
    """
    rng = np.random.default_rng(cfg.SEED)
    unique_ids = np.unique(pids)
    chosen_ids = rng.choice(unique_ids, size=min(num_ids_to_plot, len(unique_ids)),
                             replace=False)
    mask = np.isin(pids, chosen_ids)

    # Perplexity must be < number of samples; 30 is the usual default but
    # falls back gracefully for small sample counts (e.g. quick smoke tests).
    n_samples = int(mask.sum())
    perplexity = min(30, max(5, n_samples - 1))

    tsne_before = TSNE(n_components=2, random_state=cfg.SEED, init="pca",
                        perplexity=perplexity).fit_transform(embeddings_before[mask])
    tsne_after = TSNE(n_components=2, random_state=cfg.SEED, init="pca",
                       perplexity=perplexity).fit_transform(embeddings_after[mask])

    colors = plt.cm.tab20(np.linspace(0, 1, len(chosen_ids)))
    id_to_color = {pid: colors[i] for i, pid in enumerate(chosen_ids)}
    point_colors = [id_to_color[p] for p in pids[mask]]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))

    axes[0].scatter(tsne_before[:, 0], tsne_before[:, 1], c=point_colors, s=18)
    axes[0].set_title("Embeddings — no privacy (sigma = 0)")
    axes[0].set_xticks([]); axes[0].set_yticks([])

    axes[1].scatter(tsne_after[:, 0], tsne_after[:, 1], c=point_colors, s=18)
    axes[1].set_title(f"Embeddings — with privacy (sigma = {sigma})")
    axes[1].set_xticks([]); axes[1].set_yticks([])

    fig.suptitle("t-SNE: effect of Gaussian noise privacy on identity clusters")
    fig.tight_layout()

    save_path = os.path.join(cfg.FIGURES_DIR, save_name)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved figure: {save_path}")
    return save_path


def plot_privacy_utility_curve(results_by_sigma: dict, save_name: str = "privacy_utility_curve.png"):
    """
    results_by_sigma: {sigma: {"Rank-1": .., "Rank-5": .., "mAP": ..}}
    Produces the core result figure of the report: metric vs sigma.
    """
    sigmas = sorted(results_by_sigma.keys())
    metrics = ["Rank-1", "Rank-5", "mAP"]

    fig, ax = plt.subplots(figsize=(7, 5))
    for metric in metrics:
        values = [results_by_sigma[s][metric] for s in sigmas]
        ax.plot(sigmas, values, marker="o", label=metric)

    ax.set_xlabel("Noise standard deviation (sigma)")
    ax.set_ylabel("Score")
    ax.set_title("Privacy–Utility Tradeoff")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()

    save_path = os.path.join(cfg.FIGURES_DIR, save_name)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved figure: {save_path}")
    return save_path
