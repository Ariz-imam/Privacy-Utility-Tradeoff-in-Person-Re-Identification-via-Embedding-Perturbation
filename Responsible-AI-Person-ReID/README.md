Privacy–Utility Trade-off in Person Re-Identification through Embedding Perturbation

A research-oriented study investigating how embedding perturbation affects person re-identification performance and the resulting privacy–utility trade-off in AI-based video surveillance.

Motivation

Person Re-identification (Re-ID) aims to match the same individual across multiple non-overlapping camera views without relying solely on facial recognition. It plays an important role in intelligent surveillance, crowd analytics, public safety, and multi-camera tracking.

Modern Re-ID systems learn highly discriminative feature embeddings that allow reliable matching of individuals. However, these embeddings may also expose sensitive identity information if released or shared without protection.

This project investigates a simple yet interpretable privacy mechanism—Gaussian embedding perturbation—to study the trade-off between preserving identification performance and reducing identity information. Although the proposed method does not provide formal differential privacy guarantees, it serves as an empirical first step toward privacy-aware person re-identification, aligning with current Responsible AI research.

Research Question

How does person re-identification performance change as increasing amounts of Gaussian noise are injected into the learned embedding representation, and where does the privacy–utility trade-off become practically unacceptable?

Related Work

This work is inspired by several established directions in the person re-identification literature.


Market-1501 established one of the most widely used benchmark datasets and evaluation protocols for person re-identification.
Hermans et al. (2017) introduced the Batch-Hard Triplet Loss, which remains one of the standard approaches for metric learning in Re-ID.
Output perturbation using Gaussian noise has been extensively studied in privacy-preserving machine learning and motivates the embedding perturbation strategy adopted in this work.
Recent privacy-preserving person re-identification research demonstrates increasing interest in balancing surveillance utility with privacy protection. This project represents a simplified and reproducible experimental study of that broader research direction.


Methodology

Feature Extraction

The framework employs an ImageNet-pretrained ResNet-50 backbone.

The final classification layer is replaced with a 128-dimensional embedding layer followed by Batch Normalization (BNNeck) and L2 normalization to learn discriminative feature representations suitable for person re-identification.

Metric Learning

Instead of conventional classification loss, the model is trained using Batch-Hard Triplet Loss with PK sampling.

Each training batch consists of:


16 identities
4 images per identity
Batch size = 64


The objective is to minimize the distance between images of the same person while maximizing the distance between different identities.

Privacy Module

During inference, zero-mean Gaussian noise is added directly to the learned embedding vector.

After perturbation, embeddings are L2-normalized before retrieval.

The perturbation level is controlled by:

σ ∈ {0.00, 0.05, 0.10, 0.20, 0.30}

where


σ = 0.00 represents the non-private baseline
increasing σ progressively obscures identity information.


Evaluation

Performance is evaluated using the standard Market-1501 protocol.

Metrics include:


Rank-1 Accuracy
Rank-5 Accuracy
Rank-10 Accuracy
Mean Average Precision (mAP)


The privacy–utility trade-off is analyzed by evaluating retrieval performance at each perturbation level.

Dataset

Market-1501


32,668 pedestrian images
1,501 identities
6 camera views


Training set:


751 identities
12,936 images


Testing:


Query: 3,368 images
Gallery: 19,732 images


Market-1501 remains one of the most widely adopted benchmarks in person re-identification research, enabling direct comparison with prior work.

Experimental Setup

SettingValueBackboneResNet-50 (ImageNet pretrained)Embedding Dimension128Loss FunctionBatch-Hard Triplet LossMargin0.3Batch SamplingPK Sampling (16 × 4)OptimizerAdamLearning Rate3.5 × 10⁻⁴Weight Decay5 × 10⁻⁴LR ScheduleWarmup + Step DecayEpochs60Input Resolution256 × 128Data AugmentationRandom Flip, Pad-Crop, Random Erasing

Results & Discussion

Quantitative Results

Gaussian Noise (σ)Rank-1Rank-5Rank-10mAP0.0079.4%93.3%95.5%66.7%0.0570.1%90.6%94.2%54.6%0.1038.0%71.6%82.0%21.5%0.204.6%14.6%23.3%1.8%0.300.7%3.3%5.6%0.4%

Performance Analysis

The baseline model achieved 79.4% Rank-1 accuracy and 66.7% mAP, demonstrating that the ResNet-50 + Triplet Loss framework learns highly discriminative feature embeddings for person re-identification. These numbers are consistent with published triplet-loss baselines on Market-1501, which typically report Rank-1 in the 80–88% range and mAP in the 55–72% range without additional tricks such as re-ranking.

Introducing a small perturbation (σ = 0.05) reduced Rank-1 accuracy by approximately 9 percentage points (79.4% → 70.1%) and mAP by roughly 12 points (66.7% → 54.6%), while retrieval performance remained relatively strong — Rank-5 stayed above 90%. This suggests that mild embedding perturbation can obscure some identity information without severely damaging the utility of the learned representation.

Performance degradation becomes substantially more pronounced at σ = 0.10, where Rank-1 drops to 38.0% and mAP falls to 21.5% — roughly a 46% relative drop in Rank-1 and a 68% relative drop in mAP from baseline. This marks a clear "knee" in the privacy–utility curve: the region between σ = 0.05 and σ = 0.10 is where the system transitions from "usable with degraded confidence" to "substantially unreliable."

At σ = 0.20 and σ = 0.30, retrieval performance deteriorates almost completely (Rank-1 below 5%, mAP below 2%), demonstrating that strong perturbation effectively suppresses identity information but renders the representation unsuitable for reliable person matching.

Overall, the results reveal a clear monotonic privacy–utility relationship: increasing perturbation consistently reduces retrieval performance across all four metrics, illustrating the inherent trade-off between privacy preservation and identification accuracy.

Embedding Space Analysis

The accompanying t-SNE visualization provides qualitative evidence supporting the quantitative results.

Without perturbation, embeddings belonging to the same identity form compact and well-separated clusters — roughly 15 visually distinct groups are apparent even in a 2D projection. As Gaussian noise increases to σ = 0.30, these clusters collapse into a single, largely undifferentiated cloud, visually confirming the near-total loss of identity-discriminative structure reflected in the near-zero Rank-1 and mAP scores at that noise level.

Key Findings


ResNet-50 with Batch-Hard Triplet Loss provides a strong, literature-consistent baseline for person re-identification.
Moderate embedding perturbation (σ ≈ 0.05) preserves a significant portion of retrieval performance while measurably reducing identity information.
A sharp transition occurs between σ = 0.05 and σ = 0.10, beyond which retrieval performance collapses rapidly.
The study highlights the practical privacy–utility trade-off that motivates current Responsible AI research in surveillance systems.


Ablation Study

The Gaussian noise sweep serves as the primary ablation study by isolating a single variable — the perturbation magnitude — while keeping the trained model fixed. This controlled experimental design allows the privacy–utility relationship to be analyzed quantitatively rather than anecdotally.

Limitations


The proposed Gaussian perturbation is an empirical privacy mechanism and does not provide formal (ε, δ)-Differential Privacy guarantees.
Evaluation is limited to the Market-1501 benchmark.
The framework considers image-based person re-identification rather than continuous multi-camera video streams.
Privacy is evaluated indirectly through retrieval degradation rather than explicit resistance to linkage or reconstruction attacks.


Future Work

Future work may extend this study by:


integrating formally verified Differential Privacy mechanisms;
investigating adversarially trained privacy-preserving embedding networks;
evaluating transformer-based Re-ID architectures;
benchmarking across additional datasets such as DukeMTMC-ReID and MSMT17;
measuring privacy using explicit linkage and reconstruction attack success rates.
