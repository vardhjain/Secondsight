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


def test_auto_returns_a_usable_device() -> None:
    """``auto`` resolves to one of the supported device types."""
    assert resolve_device("auto").type in {"cuda", "mps", "cpu"}


def test_auto_prefers_cuda_then_mps(monkeypatch: pytest.MonkeyPatch) -> None:
    """``auto`` picks MPS when CUDA is missing, and CPU when both are."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    assert resolve_device("auto").type == "mps"
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    assert resolve_device("AUTO").type == "cpu"


def test_mps_falls_back_to_cpu_when_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    """Requesting MPS on a machine without it falls back to CPU."""
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    assert resolve_device("mps") == torch.device("cpu")


def test_cuda_is_not_silently_rerouted_to_mps(monkeypatch: pytest.MonkeyPatch) -> None:
    """An explicit CUDA request without CUDA goes to CPU even if MPS exists."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    assert resolve_device("cuda").type == "cpu"
