"""Publish the tested Gradio Space sources to an eligible free ZeroGPU account.

No credentials are stored in this repository. By default this prints a
deployment plan; --apply requires HF_TOKEN and an existing eligible account.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

FILES = ("app.py", "README.md", "requirements.txt")
SOURCE = Path(__file__).resolve().parents[1] / "space"


def publish(api: Any, repo_id: str, source: Path = SOURCE) -> str:
    """Create only a ZeroGPU Space and upload its checked-in sources."""
    if repo_id.count("/") != 1 or not all(repo_id.split("/")):
        raise ValueError("Space ID must be in namespace/name form.")
    for filename in FILES:
        if not (source / filename).is_file():
            raise ValueError(f"Space source file missing: {source / filename}")

    # No fallback to billable CPU Basic, CPU Upgrade, or a dedicated GPU.
    api.create_repo(
        repo_id=repo_id,
        repo_type="space",
        space_sdk="gradio",
        space_hardware="zero-a10g",
        private=False,
        exist_ok=True,
    )
    runtime = api.get_space_runtime(repo_id=repo_id)
    requested = str(getattr(runtime, "requested_hardware", "") or "")
    current = str(getattr(runtime, "hardware", "") or "")
    # The requested hardware is authoritative during builds/migrations.
    selected = requested or current
    if selected and not selected.startswith("zero-"):
        raise RuntimeError(
            f"Refusing to upload to hardware {selected!r}; "
            "select ZeroGPU in Space settings to avoid paid compute."
        )
    for filename in FILES:
        api.upload_file(
            path_or_fileobj=str(source / filename),
            path_in_repo=filename,
            repo_id=repo_id,
            repo_type="space",
            commit_message=f"Deploy Secondsight {filename}",
        )
    return f"https://huggingface.co/spaces/{repo_id}"


def main(argv: list[str] | None = None) -> int:
    """Preview or explicitly publish a no-cost eligible ZeroGPU Space."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", required=True, help="Hugging Face namespace/space_name")
    parser.add_argument(
        "--apply", action="store_true", help="Upload files; otherwise only preview."
    )
    args = parser.parse_args(argv)
    if args.repo_id.count("/") != 1 or not all(args.repo_id.split("/")):
        parser.error("--repo-id must have the form namespace/space_name")
    print(
        f"Space target: {args.repo_id}. Hardware: ZeroGPU only. "
        f"Files: {', '.join(FILES)}."
    )
    if not args.apply:
        print("Preview only; use --apply after preparing HF_TOKEN and checking eligibility.")
        return 0
    token = os.environ.get("HF_TOKEN")
    if not token:
        parser.error("HF_TOKEN is required for --apply (do not put tokens in commands or git).")
    from huggingface_hub import HfApi  # optional tool, not a project dependency

    url = publish(HfApi(token=token), args.repo_id)
    print(f"Upload submitted: {url}")
    print("Check the Space Build/Runtime logs and test with two authorized images.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
