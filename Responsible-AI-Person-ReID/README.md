# Privacy-Utility Tradeoff in Person Re-Identification via Embedding Perturbation

A small research-style study of how much identification performance a
person re-identification (Re-ID) system loses when its embeddings are
protected with additive Gaussian noise before release — a simple,
interpretable instance of the general privacy-preserving Re-ID problem
that motivates Responsible AI research in video surveillance.

## Motivation

Re-ID systems match the same person across non-overlapping camera
views without relying on facial recognition — useful for tracking
suspects, analyzing crowd flow, or multi-camera analytics. But the
learned embeddings a Re-ID model produces can potentially be used to
link a person's appearances across a city's camera network without
their consent, which is exactly the tension "Responsible AI" research
tries to resolve: **can a system stay useful while releasing less
identifying information than it currently does?**

## Research Question

> How much does person Re-ID accuracy degrade as we inject increasing
> amounts of Gaussian noise into the released embedding, and where is
> the point past which privacy gains stop being worth the utility
> loss?

## Related Work (brief)

- Zheng et al., *"Person Re-identification: Past, Present and
  Future"* — establishes Market-1501 and the standard CMC/mAP
  evaluation protocol used here.
- Hermans et al., *"In Defense of the Triplet Loss for Person
  Re-Identification"* (2017) — the batch-hard triplet loss and PK
  sampling strategy used for training.
- Differential-privacy-flavored output perturbation (Gaussian
  mechanism) — the general idea this project's privacy module borrows,
  applied empirically rather than as a formal DP guarantee.
- Privacy-preserving Re-ID work (e.g. WACV 2024-era papers on
  privacy-enhancing Re-ID frameworks) — the broader direction this
  project is a simplified, reproducible first step toward.

## Methodology

1. **Backbone**: ImageNet-pretrained ResNet-50, last-stage stride
   reduced to 1 (standard Re-ID modification for finer spatial
   resolution), pooled and projected to a 128-dim embedding with a
   BNNeck, then L2-normalized.
2. **Training objective**: batch-hard triplet loss (Hermans et al.,
   2017) with PK sampling (16 identities x 4 images per batch) — no
   identity classifier, no softmax. The network is trained purely to
   make same-identity embeddings closer than different-identity
   embeddings.
3. **Privacy module**: at inference, add zero-mean Gaussian noise with
   standard deviation sigma to the embedding, then re-normalize.
   sigma = 0 reproduces the non-private baseline exactly.
4. **Evaluation**: standard Market-1501 protocol — Rank-1, Rank-5,
   Rank-10 (CMC) and mAP, computed at each sigma in
   `{0.0, 0.05, 0.10, 0.20, 0.30}`.
5. **Analysis**: privacy-utility curve (metric vs sigma) and a t-SNE
   comparison of the embedding space before/after noise.

## Dataset

**Market-1501** — 32,668 images of 1,501 pedestrian identities across
6 cameras (751 identities / 12,936 images for training; 750 identities
split into a 3,368-image query set and a 19,732-image gallery for
testing). One of the most widely cited Re-ID benchmarks, which makes
results here directly comparable to published baselines. Available on
Kaggle (search "Market-1501") — see `notebooks/` for the exact input
path expected.

## Experimental Setup

| Setting | Value |
|---|---|
| Backbone | ResNet-50 (ImageNet pretrained) |
| Embedding dim | 128 |
| Loss | Batch-hard triplet, margin 0.3 |
| Sampling | PK sampling, P=16, K=4 (batch size 64) |
| Optimizer | Adam, lr 3.5e-4, weight decay 5e-4 |
| LR schedule | 5-epoch warmup, step decay x0.1 every 20 epochs |
| Epochs | 60 |
| Image size | 256 x 128 |
| Augmentation | random flip, pad+crop, random erasing |

## Results

*(Fill in after running `run_experiment.py` — the ablation table is
written to `results/privacy_utility_ablation.csv` automatically.)*

| sigma | Rank-1 | Rank-5 | Rank-10 | mAP |
|---|---|---|---|---|
| 0.00 | – | – | – | – |
| 0.05 | – | – | – | – |
| 0.10 | – | – | – | – |
| 0.20 | – | – | – | – |
| 0.30 | – | – | – | – |

See `figures/privacy_utility_curve.png` for the metric-vs-sigma plot
and `figures/tsne_comparison.png` for the embedding-space
visualization.

## Ablation Study

The sigma sweep above **is** the ablation study: it isolates the
effect of a single variable (noise magnitude) on retrieval quality
while holding the trained model fixed, which is what makes the
privacy-utility relationship measurable rather than anecdotal.

## Limitations

- The Gaussian noise mechanism is an empirical privacy proxy, not a
  formally proven (epsilon, delta)-differential-privacy guarantee —
  a natural next step would be to derive one.
- Evaluated on Market-1501 only; cross-dataset generalization
  (DukeMTMC-reID, MSMT17) is untested.
- No adversarial re-identification attack is implemented to directly
  measure "privacy gained" in attack-success-rate terms — utility loss
  is used as the (partial) proxy for privacy gained instead.

## Future Work

- Replace additive noise with a learned privacy module (e.g. an
  adversarially trained obfuscation network).
- Add a simple linkage-attack baseline to quantify privacy gain
  directly, not just utility loss.
- Extend to a formal differential-privacy analysis of the release
  mechanism.

## Repository Structure

```
Responsible-AI-Person-ReID/
├── README.md
├── requirements.txt
├── src/
│   ├── config.py         # all paths & hyperparameters
│   ├── dataset.py        # Market-1501 parsing + transforms
│   ├── sampler.py         # PK batch sampler
│   ├── model.py           # ResNet-50 embedding network
│   ├── losses.py          # batch-hard triplet loss
│   ├── privacy.py         # Gaussian noise privacy module
│   ├── train.py            # training loop
│   ├── evaluate.py         # Market-1501 CMC/mAP evaluation
│   ├── visualize.py        # t-SNE + privacy-utility curve plots
│   ├── run_experiment.py    # orchestrates train -> eval -> figures
│   └── utils.py
├── notebooks/
│   └── kaggle_responsible_ai_person_reid.ipynb   # self-contained Kaggle notebook
├── checkpoints/     # saved model weights (created at runtime)
├── results/          # CSV/JSON metrics (created at runtime)
└── figures/           # generated plots (created at runtime)
```

## How to Run (Kaggle)

1. Add the Market-1501 dataset to the notebook via **Add Data**.
2. Open `notebooks/kaggle_responsible_ai_person_reid.ipynb`, upload it
   as a new Kaggle notebook (or copy its cells in).
3. Check the dataset path printed in the first code cell matches
   `Config.DATASET_ROOT` in `config.py` — Kaggle sometimes nests the
   dataset one folder deeper than expected; the notebook prints
   `os.listdir(...)` at the top specifically so this is easy to check
   and fix before training starts.
4. Run all cells top to bottom. Enable a GPU accelerator (Settings ->
   Accelerator -> GPU) before running — training on CPU is impractically
   slow.
5. Outputs land in `/kaggle/working/{checkpoints,results,figures}`
   and can be downloaded from the notebook's Output panel.

## Citation

If reporting Market-1501 results, cite:

> Zheng, L., Shen, L., Tian, L., Wang, S., Wang, J., & Tian, Q. (2015).
> *Scalable Person Re-identification: A Benchmark.* ICCV.
