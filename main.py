"""Desktop app shell for the game jam project.

Opens a native window that renders the HTML UI in ui/ via pywebview, and
exposes a small Python API to JavaScript (window.pywebview.api.*).

Run (inside the `gamejam` conda env):
    python main.py
"""
from pathlib import Path

import webview

HTML = (Path(__file__).parent / "ui" / "index.html").resolve()


class Api:
    """Methods here are callable from JS as window.pywebview.api.<name>()."""

    def __init__(self):
        self._session_dir = None  # temp/<timestamp>/ folder for the current run

    def new_session(self):
        """Start a new run folder temp/<YYYYMMDDHHMMSS>/ that holds all its artifacts."""
        from datetime import datetime
        base = Path(__file__).parent / "temp"
        base.mkdir(exist_ok=True)
        self._session_dir = base / datetime.now().strftime("%Y%m%d%H%M%S")
        self._session_dir.mkdir(exist_ok=True)
        print(f"[session] {self._session_dir}", flush=True)
        return str(self._session_dir)

    def _temp_path(self, name):
        """Return <session-dir>/<safe-name>.png (lazily creating a session)."""
        import re
        safe = re.sub(r"[^A-Za-z0-9_-]", "_", name) or "img"
        if self._session_dir is None:
            self.new_session()
        return self._session_dir / (safe + ".png")

    def ping(self):
        return "pong from Python 🐍"

    def log(self, msg):
        """Called from JS (pylog) so UI status shows in this process's stdout."""
        print(f"[ui] {msg}", flush=True)
        return True

    def save_image(self, data_url, name):
        """Save a captured image to temp/<name>.png; return the path."""
        import base64
        import io
        from PIL import Image, ImageOps
        try:
            b64 = data_url.split(",", 1)[1]
            img = ImageOps.exif_transpose(Image.open(io.BytesIO(base64.b64decode(b64)))).convert("RGB")
            path = self._temp_path(name)
            img.save(path)
            print(f"[save] {path}", flush=True)
            return str(path)
        except Exception as e:
            print(f"[save] ERROR {e}", flush=True)
            raise

    def yolo_segment(self, data_url, fill_holes=False, device="cpu"):
        """Real-time person mask via YOLO (segmentation/yolo_segmentation.py).

        Input: a JPEG/PNG data URL of a frame. Returns a white-on-black mask PNG data URL.
        If fill_holes is set, background regions fully enclosed by the mask are filled in.
        device is 'cpu' or 'cuda' (falls back to CPU when CUDA is unavailable).
        """
        import base64
        import sys
        seg_dir = str(Path(__file__).parent / "segmentation")
        if seg_dir not in sys.path:
            sys.path.insert(0, seg_dir)
        try:
            from yolo_segmentation import segment_jpeg_to_mask_png
            raw = base64.b64decode(data_url.split(",", 1)[1])
            # loads yolo26n-seg on first call (and reloads if the device changes)
            mask_png = segment_jpeg_to_mask_png(raw, bool(fill_holes), self._resolve_device(device))
            return "data:image/png;base64," + base64.b64encode(mask_png).decode()
        except Exception as e:
            print(f"[yolo] ERROR {e}", flush=True)
            raise

    def _resolve_device(self, device):
        """Map a requested device to one we can actually use; fall back to CPU (warn once)."""
        if device == "cuda":
            try:
                import torch
                if torch.cuda.is_available():
                    return "cuda"
            except Exception:
                pass
            if not getattr(self, "_warned_cpu", False):
                print("[yolo] CUDA requested but unavailable -> using CPU", flush=True)
                self._warned_cpu = True
            return "cpu"
        return "cpu"

    def cuda_info(self):
        """Report whether the installed torch can use CUDA (for the Start-screen device picker)."""
        try:
            import torch
            available = bool(torch.cuda.is_available())
            name = torch.cuda.get_device_name(0) if available else ""
            return {"available": available, "name": name,
                    "torch": torch.__version__, "cuda": torch.version.cuda}
        except Exception as e:
            return {"available": False, "name": "", "torch": "", "cuda": None, "error": str(e)}

    def fill_mask(self, data_url):
        """Apply enclosed-hole filling to a white-on-black mask PNG; return the filled mask.

        Lets non-YOLO live backends (e.g. MediaPipe) share the exact same hole-fill, so the
        'fill enclosed holes' toggle behaves identically regardless of segmentation model.
        """
        import base64
        import sys
        import numpy as np
        import cv2
        seg_dir = str(Path(__file__).parent / "segmentation")
        if seg_dir not in sys.path:
            sys.path.insert(0, seg_dir)
        try:
            from yolo_segmentation import fill_enclosed_holes
            raw = base64.b64decode(data_url.split(",", 1)[1])
            gray = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
            _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
            ok, png = cv2.imencode(".png", fill_enclosed_holes(binary))
            if not ok:
                raise RuntimeError("Could not encode PNG")
            return "data:image/png;base64," + base64.b64encode(png.tobytes()).decode()
        except Exception as e:
            print(f"[fill_mask] ERROR {e}", flush=True)
            raise


def main():
    webview.create_window(
        "Game Jam",
        url=str(HTML),
        js_api=Api(),
        width=1000,
        height=700,
        min_size=(600, 400),
    )
    webview.start(debug=True)  # debug=True enables right-click > Inspect


if __name__ == "__main__":
    main()
