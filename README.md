# GiraffeMaxxers — Real-time Segmentation + CLIP/Emotion Scoring

A small desktop app (Python + pywebview): a two-player, split-screen webcam game.
Each side is segmented live (YOLO or MediaPipe), and each player is scored on how much their
silhouette looks like a target noun (CLIP) plus how well their face matches a target
emotion (face-api). Record for a few seconds and each player keeps their best frame.

## Requirements
- **Python 3.10–3.12** (3.11 recommended)
- **Internet**: CLIP / face-api / MediaPipe load from CDN; the YOLO model auto-downloads on first run
- Optional: an NVIDIA GPU to accelerate YOLO — see **GPU acceleration** below. Without it,
  YOLO runs on CPU (just slower) and everything still works.

## Install
With conda:
```bash
conda create -n gamejam python=3.11 -y
conda activate gamejam
pip install -r requirements.txt
```
Or with venv:
```bash
python -m venv venv
venv\Scripts\activate        # Windows (macOS/Linux: source venv/bin/activate)
pip install -r requirements.txt
```

## Run
With the environment activated:
```bash
python main.py
```
Or just double-click **`run.bat`** (Windows).

## GPU acceleration (optional)
**YOLO (segmentation)** can run on an NVIDIA GPU via CUDA — pick **CUDA (GPU)** under *YOLO device*
on the Start screen. The home screen shows whether CUDA is detected; if it isn't, the app silently
falls back to CPU.

The default install pulls a **CPU-only** PyTorch. To enable the GPU, install a CUDA build of torch
that matches your card *before* the other deps (so the CPU build isn't pulled in):
```bash
# Blackwell / RTX 50-series needs CUDA 12.8; older cards: pick cu121 etc. from https://pytorch.org
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```
Already installed the CPU build? Replace it:
```bash
pip uninstall -y torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

**Emotion (face-api) and CLIP** run in the browser (WebView2), not in Python, so they don't use
CUDA — they use the GPU through **WebGL** automatically (the console logs `face-api backend: webgl`).

## How to play
1. **Start** → on the home screen pick the segmentation backend (**YOLO** default, or **MediaPipe**),
   the **YOLO device** (CPU / CUDA), and optionally **Fill enclosed holes in mask**.
2. Two players stand in the left/right halves of the camera. Each half is segmented live,
   and each player gets three live scores:
   - **Class** — CLIP zero-shot: does your silhouette look like the target noun?
   - **Emo** — face-api: does your expression match the target emotion?
   - **Total** = `class × 2 + emotion × 1`
3. Edit the target **noun** and **emotion** at the top.
4. Set a **Record** duration → **Capture**. The app records for that long and each player
   keeps their own highest-Total frame (saved under `temp/<session>/`).

## Layout
```
main.py            App entry point (pywebview shell + Python API)
ui/                Front-end (index.html / app.js / scoring.js / style.css)
segmentation/      Segmentation helpers (yolo_segmentation.py is used by the app)
charades/          Standalone browser CLIP-charades prototypes (reference only)
requirements.txt   Dependencies
```

## Notes
- First run downloads `yolo26n-seg.pt` (~6MB) and the browser-side CLIP/face-api models (cached).
- `temp/` stores per-session images (personal data, already ignored in `.gitignore`).
