"""Shared helpers for the visualization modules.

The plotting stack (``matplotlib``, ``opencv-python-headless``,
``scikit-learn`` and ``seaborn``) ships in the optional ``viz`` extra, so every
import of those libraries goes through :func:`require`, which turns a missing
package into an :class:`ImportError` that tells the user which extra to install.
"""

from __future__ import annotations

import importlib
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover (typing only)
    from matplotlib.figure import Figure

VIZ_INSTALL_HINT = 'pip install "secondsight[viz]"'


def require(module: str, feature: str) -> ModuleType:
    """Import an optional plotting dependency or explain how to install it.

    Args:
        module: Dotted module name to import, for example ``"matplotlib.pyplot"``.
        feature: Short description of what needs the module, used in the error.

    Returns:
        The imported module.

    Raises:
        ImportError: If the module is not installed. The message names the
            ``viz`` extra that provides it.
    """
    try:
        return importlib.import_module(module)
    except ImportError as exc:
        top_level = module.split(".", maxsplit=1)[0]
        msg = f"{feature} requires {top_level}. Install it with {VIZ_INSTALL_HINT}"
        raise ImportError(msg) from exc


def pyplot() -> Any:
    """Return ``matplotlib.pyplot``, raising a helpful error when it is missing.

    Returns:
        The ``matplotlib.pyplot`` module.
    """
    return require("matplotlib.pyplot", "Plotting")


def finalize_figure(fig: Figure, save_path: str | Path | None) -> Figure:
    """Save a figure to disk and close it, or show it when no path is given.

    Args:
        fig: The matplotlib figure to finalize.
        save_path: Destination path. Parent directories are created as needed.
            When ``None`` the figure is shown with ``plt.show()`` and left open.

    Returns:
        The same figure, so callers and tests can inspect its axes.
    """
    plt = pyplot()
    if save_path is not None:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()
    return fig
