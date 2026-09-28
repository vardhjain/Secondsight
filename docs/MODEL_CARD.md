# Model Card for Secondsight

This card follows the spirit of Mitchell et al., *Model Cards for Model
Reporting* (2019). It documents a person re-identification (Re-ID) embedding
model intended for **research, education, and portfolio demonstration**.

## Model details

- **Model:** ResNet-50 + BNNeck person re-identification network that closely
  follows the "strong baseline" of Luo et al., *Bag of Tricks and a Strong
  Baseline for Deep Person Re-Identification* (CVPRW 2019), with the deliberate
  changes listed under Training procedure.
- **Version:** 0.1.0 (untagged; see the changelog for unreleased changes)
- **Task:** Deep metric learning for **cross-camera person retrieval**. The model
  maps a pedestrian image crop to a 2048-d L2-normalized embedding; identity
  matching is performed by cosine distance between embeddings (with optional
  k-reciprocal re-ranking).
- **Architecture:** ImageNet-pretrained ResNet-50 backbone with `last_stride=1`,
  generalized-mean (GeM) pooling, and a BNNeck bottleneck. A linear identity
  classifier head is used **only during training**.
- **Framework:** PyTorch (≥ 2.3).
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

- **Dataset:** Market-1501 (Zheng et al., 2015), which holds 1,501 identities
  recorded by 6 cameras outside a university supermarket. It provides 12,936
  training images across 751 identities, together with 3,368 query images of 750
  identities and a 15,913-image gallery. The gallery covers the 750 test
  identities plus 2,793 distractor crops labelled `0000`, which are kept as hard
  negatives. The 3,819 junk crops labelled `-1` in `bounding_box_test` are
  discarded on load, per the standard protocol. All crops come from an automatic
  person detector.
- **Validation:** Market-1501 has no official validation split. When
  `data.val_ids` is set, that many training identities (seen by at least two
  cameras, chosen deterministically from the seed) are held out as a small
  query and gallery set used for periodic evaluation and `best.pth` selection.
  The reported run did not use a hold-out, and its numbers come from the
  final-epoch weights rather than a test-selected checkpoint.
- **Known biases:** a single site, season, and camera rig; limited demographic
  and geographic diversity; fixed viewpoints and heights. Models trained on it
  generalize poorly across domains without adaptation.

## Training procedure

- **Sampling:** identity-balanced PK sampler (P = 16 identities × K = 4 instances
  per batch; batch size 64) so batch-hard triplet mining is well-posed.
- **Losses:** label-smoothed cross-entropy plus batch-hard triplet, with center
  loss (weight 0.0005, centers updated by SGD at learning rate 0.5) enabled for
  the reported checkpoint. All loss terms are computed in float32.
- **Optimization:** Adam with linear LR warmup followed by multistep decay at
  epochs 30 and 50, for 60 epochs; AMP mixed precision for the forward pass.
- **Deviations from Luo et al.:** GeM pooling instead of global average pooling,
  60 epochs with decay at 30 and 50 instead of 120 epochs with decay at 40 and
  70, horizontal-flip test-time augmentation, and a single seed.
- **Augmentation:** resize 256×128, horizontal flip, pad + random crop, random
  erasing, ImageNet normalization. The test-time transform is resize and
  normalize, and evaluation additionally averages each embedding with that of
  the horizontally flipped image.
- **Compute:** a single GPU (~40 minutes on a free Colab T4).

## Evaluation

- **Protocol:** single-query Market-1501. For each query, gallery images sharing
  both the query's identity **and** camera are excluded, then **CMC (Rank-k)** and
  **mean Average Precision (mAP)** are computed. Features are L2-normalized
  (cosine) with horizontal-flip test-time augmentation; **k-reciprocal
  re-ranking** is reported additionally and uses the squared Euclidean
  distance, exactly like the Zhong et al. and Bag of Tricks reference code.
- **Junk and distractors:** junk images (pid `-1`) are dropped when the dataset
  is loaded, and distractors (pid `0`) stay in the gallery as negatives.
- **Average precision:** AP is the non-interpolated mean of the precision at
  each true-match rank, as in Luo et al.'s reid-strong-baseline and torchreid.
  The official MATLAB devkit uses a trapezoidal variant that gives slightly
  different mAP.
- **Model selection:** the reported numbers come from the final-epoch weights
  evaluated once on the test split, never from a checkpoint chosen by test mAP.
- **Metrics:** measured on Market-1501 from a single training run (seed 42, 60
  epochs). The reference column lists figures reported by Luo et al. (2019) for
  the original strong-baseline recipe, for comparison only.

> **Note:** the numbers below predate the updated evaluation protocol
> (final-epoch weights and squared-distance re-ranking) and will be re-measured
> shortly. The reference figures were measured without flip test-time
> augmentation.

| Setting                   |  mAP   | Rank-1 | Rank-5 | Rank-10 | Reference (Luo et al., 2019) |
| ------------------------- | :----: | :----: | :----: | :-----: | :--------------------------: |
| Cosine + flip-TTA         | 85.04% | 94.21% | 98.25% | 98.90%  |     ~85.9 mAP / ~94.5 R-1    |
| + k-reciprocal re-ranking | 93.66% | 94.66% | 97.57% | 98.28%  |     ~94.2 mAP / ~95.4 R-1    |

## Limitations

- **Domain specificity:** trained and evaluated only on Market-1501, and
  cross-dataset transfer has not been measured for this model. The literature
  reports that direct transfer to another dataset such as MSMT17 or
  DukeMTMC-reID (which has since been withdrawn by its creators) typically loses
  ~25–30 mAP points.
- **Unmeasured demographic performance:** accuracy across age, gender, skin tone,
  body type, and attire is **not characterized**; it may be uneven.
- **Failure modes:** occlusion, low resolution, extreme lighting/viewpoint
  changes, and look-alikes (similar clothing) degrade accuracy.
- **Closed-world assumption:** the model ranks gallery candidates; it does not
  decide whether the queried person is present at all.

## Ethical considerations

Person re-identification is dual-use and surveillance-adjacent. Misuse can enable
non-consensual tracking and can disproportionately harm marginalized groups,
particularly given the unmeasured demographic performance above. Anyone building
on this work should use it only with informed consent and a lawful basis, should
never make it the sole basis for a decision about a person and should always keep
a human in the loop, should audit for demographic disparity on representative
data before any real use, and should comply with applicable privacy and
biometric-data law such as the GDPR and its equivalents.

The local Gradio demo (`app/gradio_app.py`) binds to `127.0.0.1` by default,
ships no trained weights, and supports optional authentication (`--auth`) for
any networked deployment. The optional Hugging Face Space in `space/` is
different. It is public, unauthenticated and CPU-only, compares two uploaded
crops, stores nothing, has flagging disabled, and applies an uncalibrated
cosine threshold of 0.5, so its verdict describes embedding similarity and is
not an identification. It hosts no gallery, and any example images must be
ones the Space owner has the right to publish, never Market-1501 crops.

## Caveats and recommendations

- Metrics are from a single run (seed 42); expect roughly ±0.5 mAP run-to-run
  variance. The reference column is from the literature, for comparison only.
- Re-validate on representative data before any use beyond research and
  benchmarking.
