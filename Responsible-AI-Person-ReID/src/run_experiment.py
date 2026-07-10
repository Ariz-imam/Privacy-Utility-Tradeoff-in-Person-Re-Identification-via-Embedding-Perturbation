"""
run_experiment.py
------------------
Top-level orchestration script — the single entry point that ties the
whole pipeline together:

    1. Train the embedding network (or load an existing checkpoint)
    2. Evaluate at every noise level in cfg.NOISE_SIGMA_LEVELS
    3. Save the ablation table as CSV
    4. Generate the privacy-utility curve and a t-SNE comparison figure

Run with:  python run_experiment.py
Or import `run_full_experiment()` from a notebook.
"""

import os
import json
import pandas as pd
import torch

from config import cfg
from model import build_model
from train import train
from evaluate import evaluate
from privacy import PrivacyModule
from visualize import plot_tsne_comparison, plot_privacy_utility_curve


def run_full_experiment(skip_training: bool = False, checkpoint_path: str = None,
                         num_epochs: int = None):
    if skip_training:
        assert checkpoint_path, "Provide checkpoint_path when skip_training=True"
        model = build_model()
        model.load_state_dict(torch.load(checkpoint_path, map_location=cfg.DEVICE))
        print(f"Loaded checkpoint: {checkpoint_path}")
    else:
        model, history = train(num_epochs=num_epochs)
        pd.DataFrame(history).to_csv(
            os.path.join(cfg.RESULTS_DIR, "training_history.csv"), index=False)

    model.eval()

    # ------------------------------------------------------------------
    # Ablation: evaluate at every privacy (noise) level
    # ------------------------------------------------------------------
    all_results = {}
    embeddings_by_sigma = {}
    for sigma in cfg.NOISE_SIGMA_LEVELS:
        privacy_module = PrivacyModule(sigma) if sigma > 0 else None
        results, (q_embed, q_pids, g_embed, g_pids) = evaluate(model, privacy_module)
        all_results[sigma] = results
        embeddings_by_sigma[sigma] = (g_embed.numpy(), g_pids)

    # Save ablation table
    table = pd.DataFrame(all_results).T
    table.index.name = "sigma"
    table = table.reset_index()
    table_path = os.path.join(cfg.RESULTS_DIR, "privacy_utility_ablation.csv")
    table.to_csv(table_path, index=False)
    print(f"\nSaved ablation table: {table_path}")
    print(table.to_string(index=False))

    with open(os.path.join(cfg.RESULTS_DIR, "privacy_utility_ablation.json"), "w") as f:
        json.dump(all_results, f, indent=2)

    # ------------------------------------------------------------------
    # Figures
    # ------------------------------------------------------------------
    plot_privacy_utility_curve(all_results)

    baseline_embed, baseline_pids = embeddings_by_sigma[0.0]
    highest_sigma = max(s for s in cfg.NOISE_SIGMA_LEVELS if s > 0)
    noisy_embed, _ = embeddings_by_sigma[highest_sigma]
    plot_tsne_comparison(baseline_embed, noisy_embed, baseline_pids, sigma=highest_sigma)

    return model, all_results


if __name__ == "__main__":
    run_full_experiment()
