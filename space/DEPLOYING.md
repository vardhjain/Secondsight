# Publish Secondsight's verified model demo on Hugging Face ZeroGPU

This is a **real Gradio inference demo** backed by the v0.2.0 trained model.
Two authorized person crops produce an uncalibrated embedding cosine-similarity
score, **not a same-person or different-person identity verdict**. The
Market-1501 dataset is never hosted in the Space.

## Pricing and eligibility (October 2026)

Hugging Face no longer allows every free personal account to create a new
Gradio Space on CPU Basic: a paid plan is required for ordinary Gradio CPU
Spaces. Free personal accounts **in good standing** (verified email, older
than 30 days) can instead host up to **two ZeroGPU Gradio Spaces** at no
hosting charge. Free visitors also have limited daily GPU time.

Official references:
- https://huggingface.co/docs/hub/spaces-overview
- https://huggingface.co/docs/hub/spaces-zerogpu

This guide and its deployment script request **ZeroGPU only** and do not
automatically fall back to billable resources. Check your account eligibility
before continuing. The hosted setup cannot be created or tested solely from
a GitHub connection; you must authorize your Hugging Face account.

## One-time publication

1. Ensure your Hugging Face personal account is eligible for ZeroGPU and has
   a spare free slot. Create a fine-grained **write token** for your own
   Space; keep the token out of chats, commits and logs.
2. Install the deployment dependency in your existing Python environment:
   `python -m pip install "huggingface_hub>=0.34"`.
3. Supply the token as the `HF_TOKEN` environment variable in a private
   terminal or secrets manager.
4. Preview the deployment and then explicitly publish, replacing `YOUR_HF_ID`:

```bash
python scripts/deploy_space.py --repo-id YOUR_HF_ID/Secondsight
python scripts/deploy_space.py --repo-id YOUR_HF_ID/Secondsight --apply
```

The script uploads only `space/app.py`, `space/README.md` and
`space/requirements.txt` to the Space repository, requests free
`zero-a10g` hardware and refuses known non-ZeroGPU hardware. Deployment
may fail if the account is ineligible; **do not upgrade** or supply a
payment method merely to make the portfolio demo available.

The first Space startup fetches the released `model_final.pth`
(~100.5 MB) directly from the v0.2.0 GitHub release and verifies its exact
SHA-256 and size before moving it into place. A corrupted download fails
closed. You do not need Git LFS or to upload proprietary datasets or
person images. The published model source code is pinned to v0.2.0.
The Space uses supported Python 3.12 and PyTorch 2.10.0 for ZeroGPU.

After the Space reports **RUNNING**, open its page and test an inference
request with two images you have the right to upload. Inspect the Build/
Runtime logs if checkpoint download, model loading, ZeroGPU initialization
or quota management fails. CI only verifies the code and unit tests; it
**does not** certify a hosted deployment.

## Public-facing privacy and accuracy

- Never publish Market-1501 crops as examples (research-use restrictions).
- Ask for consent before uploading pictures of other people to a public
  service. The similarity number is neither an identity match probability
  nor a biometric identification claim.
- Do not add a publicly searchable gallery or persist submitted photos.
  The app disables user flagging and example caching.
- ZeroGPU quota and hosting availability can affect public access.

## Finishing the GitHub portfolio links

After verifying a real Space URL, open your GitHub repository's **About**
gear and set the Website field to the verified Space URL. Also update the
README's demo link to the same verified URL. Until then, use the technical
case study as the Website:

https://github.com/vardhjain/Secondsight/blob/main/docs/PROJECT_SHOWCASE.md

Suggested About description (edit manually; current connector cannot edit
repository-level description/homepage):

> PyTorch person re-identification research with reproducible metric-learning
> training, Market-1501 retrieval, re-ranking, Grad-CAM, tests and CI. A
> published single run reaches 93.69% mAP and 94.95% Rank-1 with re-ranking.

Do not imply that the latest multi-seed ablations are complete until actual
benchmark runs and their provenance have been recorded.
