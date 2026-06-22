"""Desktop app shell for the game jam charades prototype."""

from pathlib import Path

import webview


HTML = (Path(__file__).parent / "ui" / "index.html").resolve()


def load_local_env():
    """Load simple KEY=VALUE lines from jam/.env without overriding the shell."""
    import os

    env_path = Path(__file__).parent / ".env"
    if not env_path.is_file():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


class Api:
    """Methods callable from JavaScript as window.pywebview.api.<name>()."""

    def __init__(self):
        self._session_dir = None

    def new_session(self):
        """Start a temp/<YYYYMMDDHHMMSS>/ folder for captured frames."""
        from datetime import datetime

        base = Path(__file__).parent / "temp"
        base.mkdir(exist_ok=True)
        self._session_dir = base / datetime.now().strftime("%Y%m%d%H%M%S")
        self._session_dir.mkdir(exist_ok=True)
        print(f"[session] {self._session_dir}", flush=True)
        return str(self._session_dir)

    def _temp_path(self, name):
        """Return <session-dir>/<safe-name>.png, lazily creating a session."""
        return self._session_file(name, ".png")

    def _session_file(self, name, suffix):
        """Return <session-dir>/<safe-name><suffix>, lazily creating a session."""
        import re

        safe = re.sub(r"[^A-Za-z0-9_-]", "_", name) or "img"
        if self._session_dir is None:
            self.new_session()
        return self._session_dir / (safe + suffix)

    def _decode_data_url_image(self, data_url):
        import base64
        import io

        from PIL import Image, ImageOps

        if not data_url:
            raise ValueError("Missing image data URL.")
        try:
            b64 = data_url.split(",", 1)[1]
        except IndexError as exc:
            raise ValueError("Invalid image data URL.") from exc
        image = ImageOps.exif_transpose(Image.open(io.BytesIO(base64.b64decode(b64))))
        image.load()
        return image

    def _prepare_flux_inputs(self, side, image_data_url, mask_data_url):
        from PIL import Image, ImageFilter

        image = self._decode_data_url_image(image_data_url).convert("RGB")
        mask = self._decode_data_url_image(mask_data_url).convert("L")

        if mask.size != image.size:
            resample = getattr(Image, "Resampling", Image).NEAREST
            mask = mask.resize(image.size, resample)
        mask = mask.point(lambda px: 255 if px >= 128 else 0)
        mask = mask.filter(ImageFilter.MaxFilter(13))

        import numpy as np
        from scipy import ndimage
        from scipy.spatial import ConvexHull
        from PIL import ImageDraw

        arr = np.array(mask)
        _, num_components = ndimage.label(arr > 0)
        if num_components > 1:
            ys, xs = np.where(arr > 0)
            points = np.column_stack([xs, ys])
            if len(points) >= 3:
                try:
                    hull = ConvexHull(points)
                    hull_pts = [(int(points[v, 0]), int(points[v, 1])) for v in hull.vertices]
                    original_mask_path = self._session_file(f"flux_{side}_mask_original", ".png")
                    mask.save(original_mask_path)
                    print(f"[flux:{side}] {num_components} components — saved original mask {original_mask_path}", flush=True)
                    ImageDraw.Draw(mask).polygon(hull_pts, fill=255)
                except Exception:
                    pass

        if mask.getbbox() is None:
            raise ValueError(f"Player {side} mask is empty.")

        image_path = self._session_file(f"flux_{side}_input", ".png")
        mask_path = self._session_file(f"flux_{side}_mask", ".png")
        image.save(image_path)
        mask.save(mask_path)
        return image_path, mask_path

    def ping(self):
        return "pong from Python"

    def log(self, msg):
        print(f"[ui] {msg}", flush=True)
        return True

    def save_image(self, data_url, name):
        """Save a captured/uploaded image to temp/<name>.png; return the path."""
        import base64
        import io

        from PIL import Image, ImageOps

        try:
            b64 = data_url.split(",", 1)[1]
            img = ImageOps.exif_transpose(
                Image.open(io.BytesIO(base64.b64decode(b64)))
            ).convert("RGB")
            path = self._temp_path(name)
            img.save(path)
            print(f"[save] {path}", flush=True)
            return str(path)
        except Exception as e:
            print(f"[save] ERROR {e}", flush=True)
            raise

    def yolo_segment(self, data_url):
        """Optional YOLO fallback. The default UI path uses browser MediaPipe."""
        import base64
        import sys

        seg_dir = str(Path(__file__).parent / "segmentation")
        if seg_dir not in sys.path:
            sys.path.insert(0, seg_dir)
        try:
            from yolo_segmentation import segment_jpeg_to_mask_png

            raw = base64.b64decode(data_url.split(",", 1)[1])
            mask_png = segment_jpeg_to_mask_png(raw)
            return "data:image/png;base64," + base64.b64encode(mask_png).decode()
        except Exception as e:
            print(f"[yolo] ERROR {e}", flush=True)
            raise

    def generate_flux_people(
        self,
        left_image_url,
        left_mask_url,
        right_image_url,
        right_mask_url,
        prompt=None,
        left_prompt=None,
        right_prompt=None,
    ):
        """Generate one BFL FLUX Fill image per player from the final best masks."""
        import base64
        from concurrent.futures import ThreadPoolExecutor, as_completed
        import os
        import traceback

        api_key = os.environ.get("BFL_API_KEY")
        if not api_key:
            return {
                "ok": False,
                "items": [],
                "error": "Missing BFL_API_KEY. Set it before launching the app.",
            }

        pairs = [
            ("L", "Player 1", left_image_url, left_mask_url),
            ("R", "Player 2", right_image_url, right_mask_url),
        ]
        pairs = [pair for pair in pairs if pair[2] and pair[3]]
        if not pairs:
            return {"ok": False, "items": [], "error": "No best image/mask pairs to inpaint."}

        try:
            from flux_inpaint import generate_inpainted_image
        except Exception as e:
            print(f"[flux] import ERROR {e}", flush=True)
            return {"ok": False, "items": [], "error": f"Could not import flux_inpaint.py: {e}"}

        default_prompt = (prompt or "replace the masked object with a photorealistic person").strip()

        prepared = []
        try:
            for side, label, image_url, mask_url in pairs:
                image_path, mask_path = self._prepare_flux_inputs(side, image_url, mask_url)
                resolved_prompt = (
                    left_prompt if side == "L" else right_prompt
                ) or default_prompt
                resolved_prompt = resolved_prompt.strip()
                prompt_path = self._session_file(f"bfl_prompt_{side}", ".txt")
                prompt_path.write_text(resolved_prompt, encoding="utf-8")
                print(f"[bfl:{side}] prompt saved {prompt_path}", flush=True)
                prepared.append(
                    {
                        "side": side,
                        "label": label,
                        "prompt": resolved_prompt,
                        "image_path": image_path,
                        "mask_path": mask_path,
                        "out_path": self._session_file(f"flux_{side}_person", ".jpg"),
                        "response_path": self._session_file(f"flux_{side}_response", ".json"),
                        "prompt_path": prompt_path,
                    }
                )
        except Exception as e:
            print(f"[flux] prepare ERROR {e}", flush=True)
            return {"ok": False, "items": [], "error": str(e)}

        def run_flux(item):
            side = item["side"]

            def status_callback(status):
                print(f"[flux:{side}] {status}", flush=True)
            print(f"[flux:{side}] generating with prompt: {item['prompt']}")
            result_path = generate_inpainted_image(
                image_path=item["image_path"],
                mask_path=item["mask_path"],
                prompt=item["prompt"],
                out_path=item["out_path"],
                api_key=api_key,
                steps=15,
                guidance=100,
                output_format="jpeg",
                safety_tolerance=6,
                poll_interval=0.75,
                timeout=300,
                save_response_json=item["response_path"],
                status_callback=status_callback,
            )
            data_url = "data:image/jpeg;base64," + base64.b64encode(result_path.read_bytes()).decode()
            return {
                "side": item["side"],
                "label": item["label"],
                "path": str(result_path),
                "mask_path": str(item["mask_path"]),
                "image_path": str(item["image_path"]),
                "prompt_path": str(item["prompt_path"]),
                "prompt": item["prompt"],
                "url": data_url,
            }

        items = []
        errors = []
        with ThreadPoolExecutor(max_workers=min(2, len(prepared))) as executor:
            futures = [executor.submit(run_flux, item) for item in prepared]
            for future in as_completed(futures):
                try:
                    items.append(future.result())
                except Exception as e:
                    print(f"[flux] generate ERROR {e}", flush=True)
                    traceback.print_exc()
                    errors.append(str(e))

        items.sort(key=lambda item: item["side"])
        return {
            "ok": bool(items),
            "items": items,
            "error": "; ".join(errors) if errors else None,
        }


def main():
    load_local_env()
    webview.create_window(
        "Game Jam",
        url=str(HTML),
        js_api=Api(),
        width=1000,
        height=700,
        min_size=(600, 400),
    )
    webview.start(debug=True)


if __name__ == "__main__":
    main()
