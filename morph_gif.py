"""morph_gif.py — turn a segmented subject into a keyword creature, then morph to it.

Pipeline
--------
1. Inputs: a segmented photo, its (hole-filled) mask, and two keywords.
2. Generate: FLUX Fill (fal.ai) paints the keywords (e.g. "angry mushroom") INTO the white
   region of the mask, so the result follows the subject's silhouette. The result is then cut
   out on a transparent background (alpha = mask).
3. Morph: build a morphing / interpolation GIF from the original subject to the generated image.

Interface
---------
    morph_gif(image_path, mask_path, keywords, out_path=None, frames=24, fps=20,
              method="dissolve", background=None, ping_pong=True, hold=6,
              max_size=1024, mask_source=True, prompt_template="{prompt}") -> out_path

`keywords` may be a list/tuple ("angry", "mushroom") or a string "angry mushroom".
Only the generation step needs the network + a fal key in <project>/.fal_key.

Run the bundled example (temp/20260619115713 pic_R + mask_R, "angry mushroom"):
    python morph_gif.py
Or with your own files:
    python morph_gif.py <image.png> <mask.png> angry mushroom
"""
from __future__ import annotations

import io
import os
import sys
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent


# ----------------------------- fal generation -----------------------------
def _ensure_fal_key():
    if os.environ.get("FAL_KEY"):
        return
    kf = ROOT / ".fal_key"
    if kf.exists():
        os.environ["FAL_KEY"] = kf.read_text(encoding="utf-8").strip()
    if not os.environ.get("FAL_KEY"):
        raise RuntimeError("FAL_KEY not set — put your fal key in .fal_key")


def _png_bytes(im: Image.Image) -> bytes:
    b = io.BytesIO()
    im.save(b, format="PNG")
    return b.getvalue()


def generate_in_mask(image: Image.Image, mask: Image.Image, prompt: str,
                     max_size: int = 1024) -> Image.Image:
    """Paint `prompt` into the white region of `mask` with FLUX Fill.

    Returns an RGBA cutout: the generated content where the mask is white, transparent
    elsewhere (the mask becomes the alpha channel).
    """
    import fal_client

    _ensure_fal_key()
    image = image.convert("RGB")
    mask = mask.convert("L").resize(image.size, Image.NEAREST)
    W, H = image.size

    # downscale for a cheaper / faster fal call (FLUX Fill is billed per call)
    scale = min(1.0, max_size / max(W, H))
    if scale < 1.0:
        gs = (round(W * scale), round(H * scale))
        img_s, mask_s = image.resize(gs, Image.LANCZOS), mask.resize(gs, Image.NEAREST)
    else:
        img_s, mask_s = image, mask

    image_url = fal_client.upload(_png_bytes(img_s), "image/png")  # uploaded from memory
    mask_url = fal_client.upload(_png_bytes(mask_s), "image/png")  # white = region to fill
    result = fal_client.subscribe(
        "fal-ai/flux-pro/v1/fill",
        arguments={"image_url": image_url, "mask_url": mask_url,
                   "prompt": prompt, "output_format": "png"},
        with_logs=False,
    )
    data = urllib.request.urlopen(result["images"][0]["url"]).read()
    gen = Image.open(io.BytesIO(data)).convert("RGB").resize(image.size, Image.LANCZOS)

    rgba = gen.convert("RGBA")
    rgba.putalpha(mask)  # cut out on a transparent background
    return rgba


# ----------------------------- morphing -----------------------------
def _smoothstep(t: float) -> float:
    return t * t * (3.0 - 2.0 * t)


def _optical_flow(a_rgb: np.ndarray, b_rgb: np.ndarray):
    """Dense flow a->b and b->a (Farneback). Returns (None, None) if cv2 is unavailable."""
    try:
        import cv2
    except Exception:
        return None, None
    ga = cv2.cvtColor(a_rgb.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    gb = cv2.cvtColor(b_rgb.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    fwd = cv2.calcOpticalFlowFarneback(ga, gb, None, 0.5, 3, 25, 3, 7, 1.5, 0)
    bwd = cv2.calcOpticalFlowFarneback(gb, ga, None, 0.5, 3, 25, 3, 7, 1.5, 0)
    return fwd, bwd


def _warp(img: np.ndarray, flow: np.ndarray, t: float) -> np.ndarray:
    import cv2
    h, w = flow.shape[:2]
    gx, gy = np.meshgrid(np.arange(w), np.arange(h))
    mapx = (gx + t * flow[..., 0]).astype(np.float32)
    mapy = (gy + t * flow[..., 1]).astype(np.float32)
    return cv2.remap(img.astype(np.float32), mapx, mapy, cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REPLICATE)


def _morph_frames(src: Image.Image, dst: Image.Image, n: int, method: str):
    """n RGBA frames interpolating src -> dst (eased)."""
    src = src.convert("RGBA")
    dst = dst.convert("RGBA").resize(src.size, Image.LANCZOS)
    a = np.asarray(src, dtype=np.float32)
    b = np.asarray(dst, dtype=np.float32)

    flow_fwd = flow_bwd = None
    if method == "flow":
        flow_fwd, flow_bwd = _optical_flow(a[..., :3], b[..., :3])

    frames = []
    for i in range(n):
        t = i / (n - 1) if n > 1 else 1.0
        tt = _smoothstep(t)
        if flow_fwd is not None:
            frame = (1 - tt) * _warp(a, flow_fwd, tt) + tt * _warp(b, flow_bwd, 1 - tt)
        else:
            frame = (1 - tt) * a + tt * b
        frames.append(Image.fromarray(np.clip(frame, 0, 255).astype(np.uint8), "RGBA"))
    return frames


# ----------------------------- GIF output -----------------------------
def _rgba_to_p_transparent(im: Image.Image) -> Image.Image:
    """RGBA -> P with index 255 as the transparent color (alpha < 128 becomes transparent)."""
    alpha = im.split()[-1]
    p = im.convert("RGB").convert("P", palette=Image.ADAPTIVE, colors=255)
    transparent = alpha.point(lambda v: 255 if v < 128 else 0)  # mask of pixels to hide
    p.paste(255, transparent)
    p.info["transparency"] = 255
    return p


def _save_gif(frames, out_path, duration_ms, background, loop, ping_pong, hold):
    seq = [frames[0]] * hold + list(frames) + [frames[-1]] * hold
    if ping_pong:
        seq = seq + seq[-2:0:-1]  # play back to the start for a seamless loop

    if background is None:  # transparent GIF
        pal = [_rgba_to_p_transparent(f) for f in seq]
        pal[0].save(out_path, save_all=True, append_images=pal[1:], duration=duration_ms,
                    loop=loop, disposal=2, transparency=255, optimize=False)
    else:  # composite over a solid color
        bg = Image.new("RGBA", frames[0].size, tuple(background) + (255,))
        flat = [Image.alpha_composite(bg, f).convert("RGB") for f in seq]
        flat[0].save(out_path, save_all=True, append_images=flat[1:], duration=duration_ms,
                     loop=loop, optimize=True)


# ----------------------------- public API -----------------------------
def morph_gif(image_path, mask_path, keywords, out_path=None, frames=24, fps=20,
              method="dissolve", background=None, ping_pong=True, hold=6,
              max_size=1024, mask_source=True, prompt_template="{prompt}"):
    """Generate a keyword creature inside the mask and morph the subject into it.

    image_path / mask_path : the segmented photo and its (hole-filled) mask.
    keywords               : ("angry", "mushroom") or "angry mushroom".
    out_path               : output .gif (defaults next to the image).
    method                 : "dissolve" (cross-fade) or "flow" (optical-flow warp morph).
    background             : None for a transparent GIF, or (r, g, b) for a solid backdrop.
    mask_source            : apply the mask to the source too (subject on transparent bg).
    Returns the gif path; also writes "<gif>_generated.png" (the transparent cutout).
    """
    image_path, mask_path = Path(image_path), Path(mask_path)
    image = Image.open(image_path).convert("RGB")
    mask = Image.open(mask_path).convert("L").resize(image.size, Image.NEAREST)

    prompt = " ".join(map(str, keywords)) if isinstance(keywords, (list, tuple)) else str(keywords)
    prompt = prompt_template.format(prompt=prompt)

    print(f"[morph] generating '{prompt}' inside the mask via FLUX Fill…", flush=True)
    generated = generate_in_mask(image, mask, prompt, max_size=max_size)

    src = image.convert("RGBA")
    if mask_source:
        src.putalpha(mask)  # subject on a transparent background, same silhouette as the target

    print(f"[morph] building {frames} frames (method={method})…", flush=True)
    frames_list = _morph_frames(src, generated, frames, method)

    if out_path is None:
        out_path = image_path.with_name(f"morph_{prompt.replace(' ', '_')}.gif")
    out_path = Path(out_path)
    gen_path = out_path.with_name(out_path.stem + "_generated.png")
    generated.save(gen_path)

    _save_gif(frames_list, str(out_path), round(1000 / fps), background, 0, ping_pong, hold)
    print(f"[morph] saved gif:       {out_path}", flush=True)
    print(f"[morph] saved cutout:    {gen_path}", flush=True)
    return str(out_path)


if __name__ == "__main__":
    if len(sys.argv) >= 5:                       # python morph_gif.py img mask kw1 kw2 ...
        morph_gif(sys.argv[1], sys.argv[2], sys.argv[3:])
    else:                                        # bundled example
        ex = ROOT / "temp" / "20260619115713"
        morph_gif(ex / "pic_R.png", ex / "mask_R.png", ("angry", "mushroom"))
