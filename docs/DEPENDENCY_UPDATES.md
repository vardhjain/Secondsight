# Updating Secondsight dependencies

## Why PyTorch and torchvision are managed as a pair

The project uses an explicit PyTorch CPU index when `--extra cpu` is selected,
but resolves ordinary PyPI builds for the default/CUDA installation. The single
`uv.lock` is a *universal* lockfile: resolution must work for the applicable
Python versions, operating systems, and optional extras, not just the developer's
current machine.

On October 26, 2026, Dependabot's weekly uv run failed while trying to upgrade
`torch` to `2.14.1+cpu` and `torchvision` to `0.29.1+cpu`. The resolver
reported an unsupported Python 3.15 / Windows lockfile split involving the
`cuda` extra. The project currently tests Python 3.10–3.14; the existing
committed lockfile still passes CI. This is an *upgrade resolution* issue, not
evidence that current installations are broken.

To keep routine updates working, `.github/dependabot.yml` excludes **version
updates only** for `torch` and `torchvision`. All other Python version updates
continue normally. GitHub Dependabot security updates remain eligible because
the exclusions use the `version-update:semver-*` update types rather than
ignoring the packages completely. Review any security advisory promptly: a
security-triggered update can still encounter the same resolver limitation.

## Update PyTorch and torchvision manually

Only perform the following when you have an upgrade to evaluate (such as
compatible upstream releases or an applicable security advisory).

1. Create a branch from the latest `main` and install the
   [uv](https://docs.astral.sh/uv/) version supported by `pyproject.toml`.
2. Update the pair **together** and regenerate the lockfile:

   ```bash
   uv lock --upgrade-package torch --upgrade-package torchvision
   ```

3. If resolution fails, examine which Python/platform/extra marker failed and
   which CPU-index and PyPI wheels are available. Do not hand-edit `uv.lock`,
   force a CUDA dependency onto the CPU index, or change the project's declared
   Python compatibility to hide the error. Retry after the needed compatible
   builds exist, or explicitly choose a compatible torch/torchvision pair.
4. Validate the lockfile and CPU install:

   ```bash
   uv lock --check
   uv sync --locked --extra cpu --extra all --extra dev
   uv run --no-sync pytest
   ```

5. If you have access to a suitable GPU environment, also test the selected
   CUDA installation and inference. A successful CPU test is **not** evidence of
   a working GPU runtime.
6. Open a PR including `uv.lock` and any manifest changes. Let CI test Python
   3.10–3.14, lint, typing, the package build, and Docker. Merge only after
   all required checks pass and the CPU/CUDA wheel matrix is understood.

## Routine weekly updates

Dependabot continues to update the other uv dependencies, the Hugging Face
Space's pip requirements, GitHub Actions, Docker, and pre-commit hooks. Those
changes are still subject to the same review and CI checks.

If Dependabot's **uv** job fails again, inspect the latest error under GitHub
**Insights → Dependency graph → Dependabot**, and distinguish (a) actual
package/index incompatibilities from (b) an existing open PR or a transient
registry failure. Revisit the exclusions if Dependabot can demonstrably upgrade
the paired CPU/CUDA lockfile without errors.
