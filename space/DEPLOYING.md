# Deploying the Secondsight demo to a Hugging Face Space

These files turn the trained model into a small public demo. A visitor uploads
two person crops and the Space reports how similar their embeddings are.

## What the demo needs

It needs only the trained weights file `best.pth`. It does not need the
Market-1501 dataset, a GPU, or any gallery images.

## Steps

1. Create a new Space at https://huggingface.co/new-space. Choose the Gradio
   SDK, give it a name such as `Secondsight`, and pick the free CPU hardware.
2. Copy the three files from this folder to the root of the Space repository,
   which are `app.py`, `requirements.txt`, and `README.md`.
3. Add the trained weights as `best.pth` at the Space root. The file is large,
   so add it with Git LFS using the commands below.

```bash
git lfs install
git lfs track "*.pth"
cp /path/to/best.pth best.pth
git add .gitattributes best.pth app.py requirements.txt README.md
git commit -m "Add Secondsight demo"
git push
```

The Space builds automatically. The first build installs PyTorch and the `reid`
package from GitHub, so it takes a few minutes. Once the Space shows as Running,
open it to try the demo.

## After it is live

Set the Space URL as the homepage in the GitHub repository's About section, the
same place the description and topics already live. That mirrors how the
reference repository links its own live demo.

## Notes

If the build runs out of disk because the default PyTorch wheel is large,
replace the two torch lines in `requirements.txt` with an exact CPU pin such as
`torch==2.4.1+cpu` and `torchvision==0.19.1+cpu`.
