# Deploying the Secondsight demo to a Hugging Face Space

These files turn the trained model into a small public demo. A visitor uploads
two person crops and the Space reports how similar their embeddings are.

## What the demo needs

It needs only a trained weights file saved as `best.pth` at the Space root. The
final-epoch weights (`outputs/model_final.pth` from training) are the ones the
published results describe, so copy that file under the name `best.pth`. The
app rebuilds the exact architecture recorded in the checkpoint, loads it with
`weights_only=True`, and never downloads ImageNet weights. It does not need the
Market-1501 dataset, a GPU, or any gallery images.

## Steps

1. Create a new Space at https://huggingface.co/new-space. Choose the Gradio
   SDK, give it a name such as `Secondsight`, and pick the free CPU hardware.
2. Copy `app.py`, `requirements.txt` and `README.md` from this folder to the
   root of the Space repository, together with an `examples/` folder if you
   want clickable example pairs (see below). The Gradio version comes from
   `sdk_version` in `README.md`, so `requirements.txt` does not pin Gradio.
3. Add the trained weights as `best.pth` at the Space root. The file is large,
   so add it with Git LFS using the commands below.

```bash
git lfs install
git lfs track "*.pth"
cp /path/to/outputs/model_final.pth best.pth
git add .gitattributes best.pth app.py requirements.txt README.md
git add examples/   # only if you added example images
git commit -m "Add Secondsight demo"
git push
```

The Space builds automatically. The first build installs PyTorch and the `reid`
package from GitHub, so it takes a few minutes. Once the Space shows as Running,
open it to try the demo.

## After it is live

Set the Space URL as the homepage in the GitHub repository's About section, the
same place the description and topics already live, and point the README's live
demo link at it. The Space also exposes a small API endpoint named `predict`.

## Sample images for the demo button

The demo shows clickable example pairs whenever an `examples/` folder is present
at the Space root. A public Space serves these files to every visitor, so only
use images you have the right to publish, such as photos of yourself or of
friends who have agreed, or synthetic renders. Never use Market-1501 crops,
because the dataset's research-use terms and this project's privacy stance rule
out redistributing them.

The images are sorted by file name and paired consecutively, so the first two
images form one example, the next two form another, and so on. Use an even
number of images, since an unpaired last image is ignored. For a clear demo,
make one pair two photos of the same person taken from different angles, and
another pair two different people. Name them so the pairs sort together, for
instance `a1.jpg`, `a2.jpg`, `b1.jpg`, `b2.jpg`, and keep the names lowercase so
the sort order is predictable. The app still runs normally when the folder is
absent.

## Notes

If the build runs out of disk because the default PyTorch wheel is large,
replace the two torch lines in `requirements.txt` with an exact CPU pin such as
`torch==2.14.0+cpu` and `torchvision==0.29.0+cpu` (keep the two versions matched).
