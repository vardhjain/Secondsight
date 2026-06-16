# Security Policy

## Supported versions

This project is in active early development. Security fixes are applied to the
latest release on the `main` branch only.

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |
| < 0.1   | :x:                |

## Reporting a vulnerability

Please **do not** report security vulnerabilities through public GitHub issues,
pull requests, or discussions.

Instead, use one of the following private channels:

1. **GitHub Security Advisories** (preferred) — open a private report via the
   repository's **Security** tab → **Report a vulnerability**.
2. **Email** — contact the maintainer at **vardhjain20@gmail.com** with the
   subject line `SECURITY: secondsight`.

When reporting, please include as much of the following as possible:

- A description of the issue and its potential impact.
- Steps to reproduce (a minimal proof of concept is ideal).
- Affected versions, environment, and any relevant configuration.

## What to expect

- **Acknowledgement** within 5 business days.
- A follow-up with an assessment and, where applicable, a remediation plan.
- Public disclosure (and credit, if desired) only after a fix is available.

## Scope and notes

This is a research/portfolio computer-vision project. A few points worth noting
for anyone deploying it:

- **Model checkpoints are untrusted input.** Loading a `.pth`/`.pt` file
  executes arbitrary code via Python's `pickle` unless `weights_only=True` is
  used. Only load checkpoints from sources you trust.
- The bundled **Gradio demo** is intended for local/offline use. It launches
  with `share=False`; do not expose it to untrusted networks without adding
  authentication and input validation appropriate to your environment.
- Dataset downloads rely on third-party services (`kagglehub`); verify their
  integrity before use.

Thank you for helping keep this project and its users safe.
