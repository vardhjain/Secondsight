# Model Card — secondsight

This card follows the spirit of Mitchell et al., *Model Cards for Model
Reporting* (2019). It documents a person re-identification (Re-ID) embedding
model intended for **research, education, and portfolio demonstration**.

## Model details

- **Model:** ResNet-50 + BNNeck person re-identification network — the "strong
  baseline" of Luo et al., *Bag of Tricks and a Strong Baseline for Deep Person
  Re-Identification* (CVPRW 2019).
- **Version:** 0.1.0
- **Task:** Deep metric learning for **cross-camera person retrieval**. The model
  maps a pedestrian image crop to a 2048-d L2-normalized embedding; identity
  matching is performed by cosine distance between embeddings (with optional
  k-reciprocal re-ranking).
- **Architecture:** ImageNet-pretrained ResNet-50 backbone with `last_stride=1`,
  generalized-mean (GeM) pooling, and a BNNeck bottleneck. A linear identity
  classifier head is used **only during training**.
- **Framework:** PyTorch (≥ 2.2).
- **License:** MIT.
- **Repository:** https://github.com/vardhjain/Secondsight
- **Contact:** vardhjain20@gmail.com

## Intended use

**Primary intended uses**

- Research and education on modern person re-identification.
- A reference implementation showcasing a production-grade CV training pipeline.
- Retrieving likely matches of a query pedestrian image within a *closed gallery*
  on the Market-1501 benchmark.

**Out-of-scope and prohibited uses**

- Real-world surveillance, tracking, or identification of specific individuals
  without their informed consent and a lawful basis.
- Any safety-, security-, or rights-critical decision about people (law
  enforcement, access control, hiring, etc.).
- Deployment on populations or camera conditions materially different from
  Market-1501 without re-validation.

The model outputs *similarity rankings*, **not** confirmed identities. It is not
a biometric identification system and must not be used as one.

## Training data

- **Dataset:** Market-1501 (Zheng et al., 2015) — 1,501 identities recorded by 6
  cameras outside a university supermarket. Splits: 12,936 train images / 751
  identities; 3,368 query and 15,913 gallery images / 750 test identities; plus
  junk/distractor crops from an automatic person detector.
- **Known biases:** a single site, season, and camera rig; limited demographic
  and geographic diversity; fixed viewpoints and heights. Models trained on it
  generalize poorly across domains without adaptation.

## Training procedure

- **Sampling:** identity-balanced PK sampler (P = 16 identities × K = 4 instances
  per batch; batch size 64) so batch-hard triplet mining is well-posed.
- **Losses:** label-smoothed cross-entropy + batch-hard triplet, with optional
  center loss.
- **Optimization:** Adam with linear LR warmup followed by multistep (or cosine)
  decay; AMP mixed precision.
- **Augmentation:** resize 256×128, horizontal flip, pad + random crop, random
  erasing, ImageNet normalization (test-time uses resize + normalize only).
- **Compute:** a single GPU (~40 minutes on a free Colab T4).

## Evaluation

- **Protocol:** single-query Market-1501. For each query, gallery images sharing
  both the query's identity **and** camera are excluded, then **CMC (Rank-k)** and
  **mean Average Precision (mAP)** are computed. Features are L2-normalized
  (cosine) with horizontal-flip test-time augmentation; **k-reciprocal
  re-ranking** is reported additionally.
- **Metrics:** measured on Market-1501 from a single training run (seed 42, 60
  epochs). The reference column lists figures reported by Luo et al. (2019) for
  the same strong baseline.

| Setting                   |  mAP   | Rank-1 | Rank-5 | Rank-10 | Reference (Luo et al., 2019) |
| ------------------------- | :----: | :----: | :----: | :-----: | :--------------------------: |
| Cosine + flip-TTA         | 85.04% | 94.21% | 98.25% | 98.90%  |     ~85.9 mAP / ~94.5 R-1    |
| + k-reciprocal re-ranking | 93.66% | 94.66% | 97.57% | 98.28%  |     ~94.2 mAP / ~95.4 R-1    |

## Limitations

- **Domain specificity:** trained and evaluated only on Market-1501. Direct
  cross-dataset transfer (e.g. to DukeMTMC-reID / MSMT17) typically loses ~25–30
  mAP points.
- **Unmeasured demographic performance:** accuracy across age, gender, skin tone,
  body type, and attire is **not characterized**; it may be uneven.
- **Failure modes:** occlusion, low resolution, extreme lighting/viewpoint
  changes, and look-alikes (similar clothing) degrade accuracy.
- **Closed-world assumption:** the model ranks gallery candidates; it does not
  decide whether the queried person is present at all.

## Ethical considerations

Person re-identification is dual-use and surveillance-adjacent. Misuse can enable
non-consensual tracking and can disproportionately harm marginalized groups,
particularly given the unmeasured demographic performance above. If you build on
this work, you should:

- use it only with informed consent and a lawful basis;
- never make it the sole basis for a decision about a person, and keep a human in
  the loop;
- audit for demographic disparity on representative data before any real use;
- comply with applicable privacy / biometric-data law (e.g. GDPR and equivalent
  statutes).

The bundled demo binds to `localhost` by default, ships no trained weights, and
supports optional authentication (`--auth`) for any networked deployment.

## Caveats and recommendations

- Metrics are from a single run (seed 42); expect roughly ±0.5 mAP run-to-run
  variance. The reference column is from the literature, for comparison only.
- Re-validate on representative data before any use beyond research and
  benchmarking.
