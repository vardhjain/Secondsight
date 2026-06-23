"""Hugging Face Space demo for Secondsight.

Upload two cropped photos of people and the app reports the cosine similarity of
their learned embeddings, plus a soft verdict on whether they are likely the
same person seen across cameras. The demo needs only the trained weights
(``best.pth``) and does not host any gallery images, so no real person imagery
is redistributed.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import gradio as gr
import torch
from PIL import Image
from torch.nn.functional import normalize
from torchvision import transforms

from reid.config import Config
from reid.models.reid_model import build_model
from reid.utils.checkpoint import infer_num_classes_from_checkpoint, load_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("secondsight.space")

WEIGHTS = Path(os.environ.get("REID_WEIGHTS", "best.pth"))
DEVICE = torch.device("cpu")
# Cosine similarity above this is reported as a likely match. It is tuned
# loosely for the demo and is not a calibrated identity decision.
SIM_THRESHOLD = 0.5

_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((256, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)


def _load_model(path: Path) -> torch.nn.Module | None:
    """Rebuild the model from the checkpoint's own config and load its weights."""
    if not path.exists():
        logger.warning("Weights %s not found; the demo will ask for them.", path)
        return None
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        config_dict = checkpoint.get("config") if isinstance(checkpoint, dict) else None
        cfg = Config.from_dict(config_dict or {})
        num_classes = infer_num_classes_from_checkpoint(path, fallback=751)
        model = build_model(cfg, num_classes=num_classes)
        load_model(model, path, map_location="cpu")
        logger.info("Loaded Secondsight weights from %s", path)
        return model.to(DEVICE).eval()
    except Exception:  # noqa: BLE001 - never crash the Space on a bad checkpoint
        logger.exception("Failed to load weights from %s", path)
        return None


_MODEL = _load_model(WEIGHTS)


@torch.no_grad()
def _embed(image: Image.Image) -> torch.Tensor:
    tensor = _TRANSFORM(image.convert("RGB")).unsqueeze(0).to(DEVICE)
    feat = _MODEL.extract_features(tensor)
    return normalize(feat.float(), dim=1)


def compare(person_a: Image.Image | None, person_b: Image.Image | None) -> str:
    """Return the cosine similarity and a soft verdict for two person crops."""
    if _MODEL is None:
        return (
            "Trained weights were not found. Add best.pth to this Space "
            "(see the deployment notes) and restart it."
        )
    if person_a is None or person_b is None:
        return "Please upload a person crop in both boxes."
    similarity = float((_embed(person_a) @ _embed(person_b).t()).item())
    verdict = "Likely the SAME person" if similarity >= SIM_THRESHOLD else "Likely DIFFERENT people"
    return (
        f"Cosine similarity: {similarity:.3f}\n"
        f"{verdict} (demo threshold {SIM_THRESHOLD:.2f})\n\n"
        "This is a research demo, not a reliable identity decision. See the model "
        "card in the GitHub repository for the limitations and intended use."
    )


_DESCRIPTION = (
    "Upload two cropped photos of people. Secondsight encodes each one with a "
    "ResNet-50 + BNNeck network trained on Market-1501 and reports how similar "
    "their embeddings are under cosine distance. A higher score means the two "
    "crops are more likely to be the same person seen on a different camera."
)

demo = gr.Interface(
    fn=compare,
    inputs=[
        gr.Image(type="pil", label="Person A"),
        gr.Image(type="pil", label="Person B"),
    ],
    outputs=gr.Textbox(label="Result", lines=5),
    title="Secondsight: cross-camera person re-identification",
    description=_DESCRIPTION,
)

if __name__ == "__main__":
    demo.launch()
