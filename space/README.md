---
title: Secondsight
emoji: 🔍
colorFrom: indigo
colorTo: blue
sdk: gradio
sdk_version: 6.28.0
python_version: "3.12"
app_file: app.py
pinned: false
license: mit
---

# Secondsight demo

Upload two cropped photos of people and this Space reports how similar their
learned embeddings are, using the ResNet-50 + BNNeck model from the
[Secondsight](https://github.com/vardhjain/Secondsight) repository trained on
Market-1501. Cosine similarity measures the alignment of two normalized
feature vectors. It does **not** estimate the probability of an identity match.

The demo is designed for eligible no-cost **ZeroGPU** hosting. The first startup
downloads the released model checkpoint from GitHub and verifies its SHA-256;
no checkpoint upload or Git LFS setup is needed. It needs no dataset and does
not host the Market-1501
gallery. It presents the raw cosine score (from -1 to +1) and **does not
classify images as the same or different person**. No identity decision
threshold has been calibrated on held-out pairs, and this model has not been
validated for real-world identification or tracking. See the GitHub repository
for the full training pipeline, measured retrieval results, and the model card
with its limitations and responsible-use guidance.
