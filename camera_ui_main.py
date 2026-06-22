"""Camera-first app shell for the game jam silhouette UI."""

from pathlib import Path

import webview

from main import Api as GameApi
from main import load_local_env


HTML = (Path(__file__).parent / "ui" / "camera_main.html").resolve()


class Api(GameApi):
    """Methods callable from JavaScript as window.pywebview.api.<name>()."""

    def __init__(self):
        super().__init__()

    def ping(self):
        return "pong from camera UI"

    def log(self, msg):
        print(f"[camera-ui] {msg}", flush=True)
        return True

    def _prepare_flux_inputs(self, side, image_data_url, mask_data_url):
        """Use the camera UI's paired black/white silhouette as the exact fill mask."""
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


def main():
    load_local_env()
    webview.create_window(
        "Game Jam Camera",
        url=str(HTML),
        js_api=Api(),
        width=1280,
        height=800,
        min_size=(900, 600),
    )
    webview.start(debug=True)


if __name__ == "__main__":
    main()
