"""
privacy.py
----------
The privacy mechanism this project studies: additive Gaussian noise on
the L2-normalized embedding before it is released for matching.

    e_private = normalize(e + N(0, sigma^2 * I))

This is a simple, well-known form of output perturbation (the same
basic idea behind Gaussian-mechanism differential privacy, though we
are NOT claiming a formal (epsilon, delta)-DP guarantee here — this is
an empirical privacy-utility study, not a formal DP proof, and that
distinction is worth stating explicitly in the report / interview).

Increasing sigma:
    - makes it harder to link two embeddings of the same person
      (protects identity / re-identification)
    - also makes it harder for the *legitimate* system to correctly
      match the same person (reduces utility)

That tradeoff, measured empirically via Rank-1 / Rank-5 / mAP as a
function of sigma, is the entire experimental core of this project.
"""

import torch
import torch.nn.functional as F


def add_gaussian_noise(embeddings: torch.Tensor, sigma: float) -> torch.Tensor:
    """
    Add zero-mean Gaussian noise with standard deviation `sigma` to a
    batch of L2-normalized embeddings, then re-normalize.

    sigma == 0.0 is a no-op and reproduces the non-private baseline
    exactly, which is what makes the ablation table's sigma=0 row a
    valid baseline rather than a special-cased branch.
    """
    if sigma <= 0.0:
        return embeddings
    noise = torch.randn_like(embeddings) * sigma
    noisy = embeddings + noise
    return F.normalize(noisy, p=2, dim=1)


class PrivacyModule:
    """Thin wrapper so evaluate.py can loop over sigma levels cleanly."""

    def __init__(self, sigma: float):
        self.sigma = sigma

    def __call__(self, embeddings: torch.Tensor) -> torch.Tensor:
        return add_gaussian_noise(embeddings, self.sigma)

    def __repr__(self):
        return f"PrivacyModule(sigma={self.sigma})"
