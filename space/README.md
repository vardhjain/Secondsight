---
title: Secondsight
emoji: 🔍
colorFrom: indigo
colorTo: blue
sdk: gradio
sdk_version: 5.50.0
python_version: "3.11"
app_file: app.py
pinned: false
license: mit
---

# Secondsight demo

Upload two cropped photos of people and this Space reports how similar their
learned embeddings are, using the ResNet-50 + BNNeck model from the
[Secondsight](https://github.com/vardhjain/Secondsight) repository trained on
Market-1501. A higher cosine similarity means the two crops are more likely to
be the same person seen on a different camera.

The demo runs on CPU, needs no dataset, and does not host any gallery images, so
no real person imagery is redistributed. See the GitHub repository for the full
training pipeline, the measured results, and the model card with its limitations
and ethics notes.
