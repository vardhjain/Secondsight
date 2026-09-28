"""Hugging Face Space demo for Secondsight.

Upload two cropped photos of people and the app reports the cosine similarity
of their learned embeddings, plus a soft verdict on whether they are likely the
same person seen across cameras. The demo needs only the trained weights
(``best.pth`` next to this file, or the path in ``REID_WEIGHTS``) and does not
host the Market-1501 gallery. Any example images come from the optional
``examples/`` folder, which should only hold photos the Space owner has the
right to publish.

Importing this module has no side effects. The model is loaded on first use
(or eagerly when the file is run as a script, which is how the Space starts
it), and the Gradio interface is built by :func:`build_demo`. The module level
``demo`` attribute is created lazily for tools such as ``gradio app.py``.
"""

from __future__ import annotations

import functools
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from torch.nn.functional import normalize

if TYPE_CHECKING:  # pragma: no cover (typing only)
    from collections.abc import Callable, Sequence

    import gradio as gr
    from PIL import Image

    from reid.models.reid_model import ReIDModel

logger = logging.getLogger("secondsight.space")

HERE = Path(__file__).resolve().parent
EXAMPLES_DIR = HERE / "examples"
_EXAMPLE_EXTS = frozenset({".jpg", ".jpeg", ".png", ".webp"})

# Cosine similarity at or above this value is reported as a likely match. The
# value is a loose demo setting and has NOT been calibrated on held-out pairs,
# so it carries no measured false-match rate and is not an identity decision.
SIM_THRESHOLD = 0.5

MISSING_WEIGHTS_MESSAGE = (
    "Trained weights were not found. Add best.pth to this Space "
    "(see the deployment notes) and restart it."
)
LOAD_FAILED_MESSAGE = "The model failed to load. Check the Space logs for the error."
_DISCLAIMER = (
    "This is a research demo, not a reliable identity decision. The threshold is "
    "uncalibrated. See the model card in the GitHub repository for the "
    "limitations and intended use."
)


@dataclass
class LoadedModel:
    """A ready-to-use embedding model and its preprocessing.

    Attributes:
        model: The Re-ID network in eval mode on the CPU.
        transform: Evaluation transform built from the checkpoint's input size.
        flip_tta: Whether to average with the horizontally flipped crop, as the
            evaluation protocol does.
    """

    model: ReIDModel
    transform: Callable[[Image.Image], torch.Tensor]
    flip_tta: bool = True


def weights_path() -> Path:
    """Return the checkpoint location, honouring ``REID_WEIGHTS``.

    Returns:
        The ``REID_WEIGHTS`` path when set, else ``best.pth`` next to this file.
    """
    return Path(os.environ.get("REID_WEIGHTS") or HERE / "best.pth")


def load(path: Path) -> tuple[LoadedModel | None, str | None]:
    """Rebuild the model from a checkpoint without any network access.

    The architecture and input size come from the config embedded in the
    checkpoint, the file is read with PyTorch's safe ``weights_only``
    unpickler, and no ImageNet weights are downloaded.

    Args:
        path: The checkpoint to load.

    Returns:
        A ``(loaded, message)`` tuple. ``loaded`` is ``None`` when the file is
        missing or fails to load, and ``message`` then explains which.
    """
    if not path.is_file():
        logger.warning("Weights %s not found; the demo will ask for them.", path)
        return None, MISSING_WEIGHTS_MESSAGE
    try:
        from reid.data.transforms import build_transforms
        from reid.models.reid_model import load_trained_model

        model, cfg = load_trained_model(path)
        transform = build_transforms(cfg.data, is_train=False)
    except Exception:  # noqa: BLE001 (never crash the Space on a bad checkpoint)
        logger.exception("Failed to load weights from %s", path)
        return None, LOAD_FAILED_MESSAGE
    logger.info("Loaded Secondsight weights from %s", path)
    return LoadedModel(model.eval(), transform, cfg.eval.flip_tta), None


@functools.cache
def get_model() -> tuple[LoadedModel | None, str | None]:
    """Load the Space's model once and cache the result.

    Returns:
        The cached result of :func:`load` for :func:`weights_path`.
    """
    return load(weights_path())


@torch.no_grad()
def embed(loaded: LoadedModel, image: Image.Image) -> torch.Tensor:
    """Embed one person crop with the evaluation protocol.

    Args:
        loaded: The model and preprocessing to use.
        image: A person crop.

    Returns:
        An L2-normalized feature tensor of shape ``(1, feat_dim)``.
    """
    tensor = loaded.transform(image.convert("RGB")).unsqueeze(0)
    feat = loaded.model.extract_features(tensor).float()
    if loaded.flip_tta:
        feat = (feat + loaded.model.extract_features(torch.flip(tensor, dims=[3])).float()) / 2
    return normalize(feat, dim=1)


def similarity_verdict(similarity: float, threshold: float = SIM_THRESHOLD) -> str:
    """Turn a cosine similarity into the demo's soft verdict.

    Args:
        similarity: Cosine similarity of two embeddings.
        threshold: Similarity at or above which the pair is called a match.

    Returns:
        A short human-readable verdict.
    """
    return "Likely the SAME person" if similarity >= threshold else "Likely DIFFERENT people"


def compare_with(
    loaded: LoadedModel | None,
    person_a: Image.Image | None,
    person_b: Image.Image | None,
    error: str | None = None,
) -> str:
    """Compare two crops with an explicit model, for the UI and for tests.

    Args:
        loaded: The model to use, or ``None`` when it is unavailable.
        person_a: First crop, or ``None`` when not uploaded.
        person_b: Second crop, or ``None`` when not uploaded.
        error: Message explaining why ``loaded`` is ``None``.

    Returns:
        The text shown in the result box.
    """
    if loaded is None:
        return error or MISSING_WEIGHTS_MESSAGE
    if person_a is None or person_b is None:
        return "Please upload a person crop in both boxes."
    similarity = float((embed(loaded, person_a) @ embed(loaded, person_b).t()).item())
    return (
        f"Cosine similarity: {similarity:.3f}\n"
        f"{similarity_verdict(similarity)} (demo threshold {SIM_THRESHOLD:.2f})\n\n"
        f"{_DISCLAIMER}"
    )


def compare(person_a: Image.Image | None, person_b: Image.Image | None) -> str:
    """Return the cosine similarity and a soft verdict for two person crops.

    Args:
        person_a: First crop, or ``None`` when not uploaded.
        person_b: Second crop, or ``None`` when not uploaded.

    Returns:
        The text shown in the result box.
    """
    loaded, error = get_model()
    return compare_with(loaded, person_a, person_b, error)


def pair_examples(paths: Sequence[str]) -> list[list[str]] | None:
    """Pair example images consecutively (first with second, and so on).

    Args:
        paths: Image paths in display order.

    Returns:
        The list of pairs, or ``None`` when there is no complete pair. An
        unpaired last image is dropped with a logged warning.
    """
    if len(paths) % 2:
        logger.warning("Ignoring unpaired example image %s", paths[-1])
    pairs = [[paths[i], paths[i + 1]] for i in range(0, len(paths) - 1, 2)]
    return pairs or None


def find_examples(directory: Path = EXAMPLES_DIR) -> list[list[str]] | None:
    """Collect example pairs from a folder of images sorted by file name.

    Args:
        directory: Folder to scan. It may be missing.

    Returns:
        Example pairs for :class:`gradio.Interface`, or ``None``.
    """
    if not directory.is_dir():
        return None
    images = sorted(
        (p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in _EXAMPLE_EXTS),
        key=lambda p: p.name.lower(),
    )
    return pair_examples([str(p) for p in images])


def build_demo(examples: list[list[str]] | None = None) -> gr.Interface:
    """Build the Gradio interface.

    Args:
        examples: Example pairs. Defaults to the pairs found in ``examples/``.

    Returns:
        The :class:`gradio.Interface`, ready to launch.
    """
    import gradio as gr

    if examples is None:
        examples = find_examples()
    lead = (
        "Upload two cropped photos of people, or click an example below."
        if examples
        else "Upload two cropped photos of people."
    )
    description = (
        f"{lead} Secondsight encodes each one with a ResNet-50 + BNNeck network "
        "trained on Market-1501 and reports how similar their embeddings are under "
        "cosine distance. A higher score means the two crops are more likely to be "
        "the same person seen on a different camera."
    )
    return gr.Interface(
        fn=compare,
        inputs=[
            gr.Image(type="pil", label="Person A"),
            gr.Image(type="pil", label="Person B"),
        ],
        outputs=gr.Textbox(label="Result", lines=5),
        title="Secondsight: cross-camera person re-identification",
        description=description,
        examples=examples,
        cache_examples=False,
        flagging_mode="never",
        api_name="predict",
    )


def __getattr__(name: str) -> Any:
    """Build the module level ``demo`` lazily for ``gradio app.py`` reload mode.

    Args:
        name: The attribute being looked up.

    Returns:
        The cached interface when ``name`` is ``"demo"``.

    Raises:
        AttributeError: For any other missing attribute.
    """
    if name == "demo":
        demo = build_demo()
        globals()["demo"] = demo
        return demo
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    get_model()  # load at start-up so problems show in the logs immediately
    build_demo().launch()
