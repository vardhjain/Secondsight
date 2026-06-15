"""Tests for device resolution (:mod:`reid.utils.device`)."""

from __future__ import annotations

import pytest
import torch

from reid.utils.device import resolve_device


def test_cpu_is_resolved_directly() -> None:
    """A requested ``cpu`` device is returned unchanged."""
    assert resolve_device("cpu") == torch.device("cpu")


@pytest.mark.skipif(torch.cuda.is_available(), reason="CUDA is available, so no fallback occurs.")
@pytest.mark.parametrize("requested", ["cuda", "cuda:0"])
def test_cuda_falls_back_to_cpu_when_unavailable(requested: str) -> None:
    """Requesting CUDA without a CUDA device falls back to CPU."""
    assert resolve_device(requested).type == "cpu"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device present.")
def test_cuda_is_resolved_when_available() -> None:
    """A requested CUDA device is honored when CUDA is available."""
    assert resolve_device("cuda").type == "cuda"
