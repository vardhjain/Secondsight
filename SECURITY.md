# Security Policy

## Supported versions

This project is in active early development and has not published a tagged
release yet. Security fixes are applied to the `main` branch only.

| Version            | Supported          |
| ------------------ | ------------------ |
| `main` (latest)    | :white_check_mark: |
| older commits      | :x:                |

## Reporting a vulnerability

Please **do not** report security vulnerabilities through public GitHub issues,
pull requests, or discussions.

Instead, please report it privately through one of the two channels below.

1. **GitHub Security Advisories** (preferred). Open a private report from the
   repository's **Security** tab using **Report a vulnerability**.
2. **Email**. Write to the maintainer at **vardhjain20@gmail.com** with the
   subject line `SECURITY: Secondsight`.

When you report an issue, please include as much of the following as you can.

- A description of the issue and its potential impact.
- Steps to reproduce (a minimal proof of concept is ideal).
- Affected versions, environment, and any relevant configuration.

## What to expect

- **Acknowledgement** within 5 business days.
- A follow-up with an assessment and, where applicable, a remediation plan.
- Public disclosure (and credit, if desired) only after a fix is available.

## Scope and notes

This is a research/portfolio computer-vision project. Anyone deploying it should
keep the following points in mind.

- **Model checkpoints are untrusted input.** Loading a `.pth`/`.pt` file
  executes arbitrary code via Python's `pickle` unless `weights_only=True` is
  used. The bundled evaluation, visualization and demo paths load with
  `weights_only=True`, but you should still only load checkpoints from sources
  you trust.
- The local **Gradio demo** (`app/gradio_app.py`) is intended for local or
  offline use. It launches with `share=False` and binds to `127.0.0.1` unless
  `--server-name` or `GRADIO_SERVER_NAME` says otherwise (the Docker image sets
  it to `0.0.0.0` so the mapped port works). Do not expose it to untrusted
  networks without adding authentication (`--auth`) and input validation
  appropriate to your environment.
- The optional **Hugging Face Space** (`space/`) is a public, unauthenticated
  CPU demo that compares two uploaded crops. It stores no uploads, has flagging
  disabled, and reports an uncalibrated similarity, so it must not be treated
  as an identification service.
- Dataset downloads rely on third-party services (`kagglehub`); verify their
  integrity before use.

Thank you for helping keep this project and its users safe.
