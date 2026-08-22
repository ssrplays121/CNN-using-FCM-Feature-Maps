# CNN using FCM Feature Maps

A convolutional network for handwritten digit classification (0–9) that adds a **Fuzzy C-Means (FCM) attention layer** on top of its convolutional feature maps — plus a companion notebook that opens the whole pipeline up and shows exactly what that layer is doing, step by step, with hand-crafted kernels and real numbers.

## Why FCM attention?

A standard CNN pools its final conv feature maps straight into fully-connected layers, treating every spatial location as equally important. Here, before that happens, an `FCMLayer` runs **fuzzy c-means clustering** over the spatial locations of the last conv feature map, treating each location's stack of channel activations as a point in feature space. Locations that cluster together strongly get a high "membership" score; that score becomes a soft attention map which reweights the feature map before it's flattened — nudging the network to focus on the spatial regions that are most distinctive, without hard-assigning any region to a single cluster.

## Repository structure

```
.
├── .devcontainer/
│   ├── devcontainer.json      # devcontainer config: base image, env wiring, VS Code settings
│   ├── .env.example           # template for local secrets — copy to .env and fill in
│   └── .env                   # (git-ignored) your real Kaggle credentials
├── sample-digits/              # 1 sample image per class, used only by the visualization notebook
│   ├── 0.jpg
│   ├── 1.jpg
│   ├── ...
│   └── 9.jpg
├── fcm-cnn-digits.ipynb              # trains the FCM_CNN model
├── fcm_cnn_visualization.ipynb       # visualizes the FCM attention mechanism, no training
└── README.md
```

> The `sample-digits/` path above is a suggestion — if it ends up somewhere else in your checkout, just update the `DATA_DIR` variable near the top of `fcm_cnn_visualization.ipynb` to match.

## The notebooks

### 1. `fcm-cnn-digits.ipynb` — training

Trains `FCM_CNN` end-to-end on a handwritten-digits (0–9) image dataset pulled via `torchvision.datasets.ImageFolder`.

**Architecture:**

| Layer | Operation | Output shape |
|---|---|---|
| `conv1` | 3×3 conv, padding 1 | 64 × 28 × 28 |
| `pool1` | 2×2 max pool | 64 × 14 × 14 |
| `conv2` | 5×5 conv, padding 2 | 128 × 14 × 14 |
| `pool2` | 2×2 max pool | 128 × 7 × 7 |
| `conv3` | 7×7 conv, padding 3 | 256 × 7 × 7 |
| `FCMLayer` | fuzzy c-means over the 49 spatial locations → attention map `A` | 1 × 7 × 7 |
| — | `x_refined = x * A` (elementwise) | 256 × 7 × 7 |
| `fc1`, `fc2` | flatten → dense → dense | 10 classes |

`FCMLayer` initializes `K` random spatial locations as cluster centers, then alternates computing fuzzy memberships and updating centers for a fixed number of iterations — the same distance → membership → center-update loop from classic fuzzy c-means, just running inside the forward pass on GPU/CPU tensors. The final attention map is the max membership across clusters at each location.

Runs for 10 epochs, printing training loss and validation accuracy per epoch.

### 2. `fcm_cnn_visualization.ipynb` — visualization companion

A **pure visualization notebook — no training, no learned weights.** Every kernel is hand-crafted so that each intermediate feature map is something you can actually look at and interpret, and every operation (convolution, max pooling, fuzzy c-means) is implemented from scratch in NumPy so the underlying math is fully exposed.

It walks through the same spatial-size progression as the real model (28 → 14 → 7, with matching padding) on a single sample image at a time:

1. **Load → grayscale → resize to 28×28**
2. **3×3 conv → 1 feature map** (hand-made edge kernel) → **2×2 max pool** (28→14)
3. **5×5 conv → 2 feature maps** (horizontal- and vertical-edge kernels) → **2×2 max pool** (14→7)
4. **7×7 conv → 4 feature maps** (Gabor kernels at 0°/45°/90°/135°, no pooling) — these 4 maps are exactly what feeds the FCM layer
5. **Inside the FCM layer**, shown step by step:
   - reshape the 4 feature maps into 49 spatial feature vectors
   - randomly initialize cluster centers (plotted on the feature map)
   - iterate {distance → fuzzy membership → center update}, with the first iteration fully worked out numerically
   - visualize membership maps evolving from iteration 1 to the final iteration
   - build the attention map `A` (max membership per location)
   - refine the 4 feature maps via `x * A`, shown before/after

Every formula (convolution, max pooling, distance, membership, center update) is followed by a printed worked example using real numbers from the chosen image — patch, kernel, product, and result — not just the equation.

**Interactive controls**, exposed as widgets at the top of the notebook:
- which digit (`0`–`9`) to visualize, pulled from `sample-digits/`
- FCM fuzziness `m`
- number of FCM iterations
- number of clusters `K`

Change any of these and re-run the cells below to see the effect ripple through the pipeline.

## Sample data

`sample-digits/` holds exactly one image per class (`0.jpg` through `9.jpg`), used solely by the visualization notebook so it has something concrete to run its hand-crafted pipeline on. It is **not** used for training — `fcm-cnn-digits.ipynb` pulls its own, much larger dataset separately.

## Dev container setup

This repo includes a `.devcontainer/devcontainer.json` that launches straight from **Kaggle's official Python Docker image** (`gcr.io/kaggle-images/python:latest`) — the same image that backs Kaggle Notebooks/Kernels — so the local environment matches Kaggle's exactly: PyTorch, torchvision, NumPy, pandas, scikit-learn, Jupyter, `ipywidgets`, and `kagglehub` all come pre-installed, with no Dockerfile and no extra build step beyond pulling that (large) image.

```json
{
    "name": "Kaggle DevContainer",
    "image": "gcr.io/kaggle-images/python:latest",
    "runArgs": [
        "--env-file",
        ".devcontainer/.env"
    ],
    "customizations": {
        "vscode": {
            "settings": {
                "python.defaultInterpreterPath": "/opt/conda/bin/python"
            },
            "extensions": [
                "ms-python.python",
                "ms-toolsai.jupyter"
            ]
        }
    },
    "remoteUser": "root"
}
```

What each piece does:
- **`image`** — pulls the Kaggle image directly; no custom build.
- **`runArgs: ["--env-file", ".devcontainer/.env"]`** — passes Docker's native `--env-file` flag, loading every variable in `.devcontainer/.env` (`KAGGLE_USERNAME`, `KAGGLE_KEY`) as container environment variables at startup. `kagglehub` picks these up automatically on first use — no interactive login, no manually-placed `kaggle.json`.
- **`customizations.vscode.settings`** — points VS Code's Python interpreter at `/opt/conda/bin/python`, the conda environment baked into the Kaggle image.
- **`customizations.vscode.extensions`** — auto-installs the Python and Jupyter extensions inside the container.
- **`remoteUser: "root"`** — the container runs as root, matching how the Kaggle image is normally used.

**Getting started:**

1. Grab a Kaggle API token: on kaggle.com go to **Settings → API → Create New Token**. This downloads a `kaggle.json` containing your `username` and `key`.
2. Copy the env template and fill it in, inside `.devcontainer/`:
   ```bash
   cp .devcontainer/.env.example .devcontainer/.env
   ```
   then edit `.devcontainer/.env`:
   ```
   KAGGLE_USERNAME=your_kaggle_username
   KAGGLE_KEY=your_kaggle_key
   ```
3. Open the repo in VS Code with the *Dev Containers* extension (or GitHub Codespaces, or the `devcontainer` CLI) and choose **Reopen in Container** — it will pick up `.devcontainer/devcontainer.json` automatically.
4. The first build pulls the Kaggle base image, which is large (it bundles the full data-science/ML stack), so expect the initial pull to take a few minutes. It's cached after that, so subsequent opens are fast.
5. Once the container is up, open either notebook — the same environment serves both, so `fcm-cnn-digits.ipynb` (training, needs the Kaggle dataset via `kagglehub`) and `fcm_cnn_visualization.ipynb` (visualization, only needs `sample-digits/`) both run without any additional installs.

**A note on secrets:** never commit `.devcontainer/.env` or a real `kaggle.json`. Only `.devcontainer/.env.example`, with blank placeholder values, should be tracked in version control.

## Requirements

Everything needed is pre-installed in the dev container. If you're running outside of it, you'll need at minimum: `torch`, `torchvision`, `numpy`, `matplotlib`, `Pillow`, `ipywidgets`, and `kagglehub` (only required for the training notebook's dataset download).