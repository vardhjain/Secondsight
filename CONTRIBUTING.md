# Contributing

Thanks for your interest in improving **Secondsight**! This document
explains how to set up a development environment and the conventions the project
follows.

## Development setup

The project supports **Python 3.10 through 3.14** and uses [`uv`](https://github.com/astral-sh/uv)
for environment and dependency management.

```bash
# 1. Clone
git clone https://github.com/vardhjain/Secondsight.git
cd Secondsight

# 2. Create the environment and install the package, all extras and dev tools
uv sync --extra all --extra dev               # or: make install
#    On a machine without a CUDA GPU, add --extra cpu to get the CPU PyTorch wheels

# 3. Install the pre-commit hooks
uv run pre-commit install
```

If you prefer plain `pip`, install the package in editable mode.

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[all,dev]"
```

## Quality gates

Every change must pass the same checks CI runs, and the `Makefile` wraps them.

```bash
make format        # ruff format . && ruff check --fix .
make lint          # ruff check .
make format-check  # ruff format --check . && ruff check .   (what CI enforces)
make typecheck     # mypy src/reid
make test          # pytest
```

- **Formatting & linting:** [ruff](https://docs.astral.sh/ruff/) (line length 100,
  Google-style docstrings). Run `make format` before committing.
- **Types:** public functions and classes are fully type-annotated; `mypy` runs
  in non-blocking mode in CI.
- **Tests:** [pytest](https://docs.pytest.org/). Light tests run on CPU with only
  NumPy, PyTorch, Pillow and PyYAML; heavy tests (the model and the transform pipeline)
  are guarded with `pytest.importorskip("torchvision")` and skip cleanly when
  `torchvision` is absent. Tests must never touch the network (a shared fixture
  blocks downloads), and new behavior should come with a test.

## Conventions

- **Style:** keep modules small and single-purpose; prefer pure functions and
  dependency injection over globals. Avoid `print` inside the library, and use the
  `logging` module (`reid.utils.logging.setup_logger`).
- **Lazy heavy imports:** importing the top-level `reid` package (and the
  numpy-only `reid.evaluation.metrics`) must never require an optional extra.
  Import matplotlib, seaborn, scikit-learn, OpenCV, Gradio and kagglehub inside
  the function that needs them, and when one is missing raise an `ImportError`
  that names the extra, for example `pip install "secondsight[viz]"`.
- **Configuration:** new knobs go through `reid.config` dataclasses and the YAML
  files, never as hardcoded constants in the training/eval paths.
- **Commits:** clear, imperative subject lines (e.g. "Add cosine scheduler").
  [Conventional Commits](https://www.conventionalcommits.org/) prefixes
  (`feat:`, `fix:`, `docs:`, …) are welcome but not required.

## Pull requests

1. Branch off `main`.
2. Make your change with tests and docs.
3. Ensure `make format-check`, `make typecheck`, and `make test` all pass.
4. Open a PR using the template; describe the motivation and any results.

By contributing you agree that your contributions are licensed under the
project's [MIT License](LICENSE) and that you will uphold the
[Code of Conduct](CODE_OF_CONDUCT.md).
