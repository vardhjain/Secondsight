# Contributing

Thanks for your interest in improving **secondsight**! This document
explains how to set up a development environment and the conventions the project
follows.

## Development setup

The project targets **Python 3.10+** and uses [`uv`](https://github.com/astral-sh/uv)
for environment and dependency management.

```bash
# 1. Clone
git clone https://github.com/vardhjain/secondsight.git
cd secondsight

# 2. Create the environment and install the package + dev tools (editable)
uv sync --extra dev          # or: make install

# 3. Install the pre-commit hooks
uv run pre-commit install
```

If you prefer plain `pip`:

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

## Quality gates

Every change must pass the same checks CI runs. The `Makefile` wraps them:

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
  `numpy`/`torch`/`Pillow`; heavy tests (the model and the transform pipeline)
  are guarded with `pytest.importorskip("torchvision")` and skip cleanly when
  `torchvision` is absent. New behavior should come with a test.

## Conventions

- **Style:** keep modules small and single-purpose; prefer pure functions and
  dependency injection over globals. No `print` inside the library — use the
  `logging` module (`reid.utils.logging.setup_logger`).
- **Lazy heavy imports:** importing the top-level `reid` package (and the
  numpy-only `reid.evaluation.metrics`) must never require `torchvision`,
  `cv2`, `gradio`, or `kagglehub`. Import heavy optional deps inside the
  submodule/function that needs them.
- **Configuration:** new knobs go through `reid.config` dataclasses and the YAML
  files — avoid hardcoded constants in the training/eval paths.
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
