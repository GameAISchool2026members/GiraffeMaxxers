#!/usr/bin/env python3
"""Run FLUX.1 Fill inpainting with an original image and black/white mask.

The mask follows the FLUX.1 Fill convention:
  - black pixels are preserved
  - white pixels are inpainted

Import from another script:
    from flux_inpaint import generate_inpainted_image

    result_path = generate_inpainted_image(
        image_path="images/original.jpg",
        mask_path="images/mask.png",
        prompt="replace the masked object with a red backpack",
        out_path="outputs/flux_inpainted.jpg",
    )

Example:
    python flux_inpaint.py ^
        --image images/original.jpg ^
        --mask images/mask.png ^
        --prompt "replace the masked object with a red backpack" ^
        --out outputs/flux_inpainted.jpg
"""

from __future__ import annotations

import argparse
import base64
from datetime import datetime
import json
import os
import time
from pathlib import Path
from typing import Any, Callable, Optional

import requests

try:
    from PIL import Image
except ImportError:
    Image = None


API_URL = "https://api.bfl.ai/v1/flux-pro-1.0-fill"
READY_STATUS = "Ready"
FAILED_STATUSES = {"Error", "Failed"}
DEFAULT_OUTPUT_DIR = Path("outputs")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inpaint an image with FLUX.1 Fill using a separate black/white mask.",
    )
    parser.add_argument("--image", required=True, type=Path, help="Original image path.")
    parser.add_argument("--mask", required=True, type=Path, help="Black/white mask path. White is edited.")
    parser.add_argument("--prompt", required=True, help="Text prompt describing what to generate in the mask.")
    parser.add_argument(
        "--out",
        type=Path,
        help="Where to save the generated image. Defaults to outputs/flux_inpainted_<timestamp>.<ext>.",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("BFL_API_KEY"),
        help="BFL API key. Defaults to the BFL_API_KEY environment variable.",
    )
    parser.add_argument(
        "--steps",
        default=50,
        type=int,
        help="Generation steps sent to FLUX.1 Fill.",
    )
    parser.add_argument(
        "--guidance",
        default=100,
        type=float,
        help="Prompt guidance value sent to FLUX.1 Fill.",
    )
    parser.add_argument(
        "--output-format",
        choices=("jpeg", "png"),
        default=None,
        help="Requested API output format. Defaults to the extension of --out, or jpeg.",
    )
    parser.add_argument(
        "--safety-tolerance",
        default=6,
        type=int,
        help="Safety tolerance value sent to FLUX.1 Fill.",
    )
    parser.add_argument(
        "--poll-interval",
        default=0.5,
        type=float,
        help="Seconds to wait between polling attempts.",
    )
    parser.add_argument(
        "--timeout",
        default=300,
        type=float,
        help="Maximum seconds to wait for generation.",
    )
    parser.add_argument(
        "--save-response-json",
        type=Path,
        help="Optional path to save the final API polling response JSON.",
    )
    return parser.parse_args()


def encode_file(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"File does not exist: {path}")
    return base64.b64encode(path.read_bytes()).decode("utf-8")


def validate_image_and_mask(image_path: Path, mask_path: Path) -> None:
    """Catch common input issues before the API returns a 422 validation error."""
    if not image_path.is_file():
        raise FileNotFoundError(f"Image file does not exist: {image_path}")
    if not mask_path.is_file():
        raise FileNotFoundError(f"Mask file does not exist: {mask_path}")
    if Image is None:
        return

    with Image.open(image_path) as image, Image.open(mask_path) as mask:
        if image.size != mask.size:
            raise ValueError(
                "Image and mask must have the same dimensions. "
                f"Got image {image.size[0]}x{image.size[1]} and "
                f"mask {mask.size[0]}x{mask.size[1]}."
            )


def sanitize_api_detail(value: Any) -> Any:
    """Remove large/base64 request payloads from provider validation errors."""
    if isinstance(value, dict):
        sanitized = {}
        for key, item in value.items():
            if key in {"image", "mask", "image_url", "mask_url"}:
                sanitized[key] = "<redacted>"
            else:
                sanitized[key] = sanitize_api_detail(item)
        return sanitized
    if isinstance(value, list):
        return [sanitize_api_detail(item) for item in value]
    if isinstance(value, str) and len(value) > 240:
        return value[:120] + "...<redacted>..." + value[-40:]
    return value


def raise_for_api_error(response: requests.Response, context: str) -> None:
    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        try:
            detail = sanitize_api_detail(response.json())
        except ValueError:
            detail = sanitize_api_detail(response.text)
        raise RuntimeError(
            f"{context} failed with HTTP {response.status_code}: {detail}"
        ) from exc


def infer_output_format(out_path: Optional[Path]) -> str:
    if out_path is None:
        return "jpeg"
    if out_path.suffix.lower() == ".png":
        return "png"
    return "jpeg"


def make_default_output_path(output_format: str) -> Path:
    suffix = ".png" if output_format == "png" else ".jpg"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return DEFAULT_OUTPUT_DIR / f"flux_inpainted_{timestamp}{suffix}"


def create_request(
    image_path: Path,
    mask_path: Path,
    prompt: str,
    api_key: str,
    *,
    steps: int = 50,
    guidance: float = 60,
    output_format: str = "jpeg",
    safety_tolerance: int = 6,
) -> tuple[str, str]:
    headers = {
        "x-key": api_key,
        "Content-Type": "application/json",
    }
    payload = {
        "prompt": prompt,
        "image": encode_file(image_path),
        "mask": encode_file(mask_path),
        "steps": steps,
        "guidance": guidance,
        "output_format": output_format,
        "safety_tolerance": safety_tolerance,
    }

    response = requests.post(API_URL, headers=headers, json=payload, timeout=60)
    raise_for_api_error(response, "Creating FLUX.1 Fill request")
    data = response.json()

    try:
        return data["id"], data["polling_url"]
    except KeyError as exc:
        raise RuntimeError(f"Unexpected create response: {data}") from exc


def poll_result(
    polling_url: str,
    api_key: str,
    poll_interval: float,
    timeout: float,
    status_callback: Optional[Callable[[Optional[str]], None]] = None,
) -> dict[str, Any]:
    headers = {
        "accept": "application/json",
        "x-key": api_key,
    }
    deadline = time.monotonic() + timeout

    while time.monotonic() < deadline:
        response = requests.get(polling_url, headers=headers, timeout=60)
        raise_for_api_error(response, "Polling FLUX.1 Fill result")
        data = response.json()
        status = data.get("status")

        if status_callback:
            status_callback(status)
        if status == READY_STATUS:
            return data
        if status in FAILED_STATUSES:
            raise RuntimeError(f"Generation failed: {data}")

        time.sleep(poll_interval)

    raise TimeoutError(f"Timed out after {timeout:g} seconds waiting for FLUX.1 Fill.")


def download_result(result_url: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with requests.get(result_url, stream=True, timeout=120) as response:
        raise_for_api_error(response, "Downloading FLUX.1 Fill result")
        with out_path.open("wb") as file:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    file.write(chunk)


def save_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def get_result_url(result: dict[str, Any]) -> str:
    try:
        return result["result"]["sample"]
    except KeyError as exc:
        raise RuntimeError(f"Ready response did not include result.sample: {result}") from exc


def generate_inpainted_image(
    image_path: str | Path,
    mask_path: str | Path,
    prompt: str,
    out_path: Optional[str | Path] = None,
    *,
    api_key: Optional[str] = None,
    steps: int = 50,
    guidance: float = 60,
    output_format: Optional[str] = None,
    safety_tolerance: int = 6,
    poll_interval: float = 0.5,
    timeout: float = 300,
    save_response_json: Optional[str | Path] = None,
    status_callback: Optional[Callable[[Optional[str]], None]] = None,
) -> Path:
    """Generate an inpainted image, save it, and return the saved image path.

    Args:
        image_path: Original image file.
        mask_path: Black/white mask file. White areas are edited.
        prompt: Text prompt describing what to generate inside the mask.
        out_path: Optional output path. If omitted, an outputs/ filename is created.
        api_key: Optional BFL API key. Defaults to the BFL_API_KEY environment variable.
        status_callback: Optional function called with each polling status.
    """
    resolved_api_key = api_key or os.environ.get("BFL_API_KEY")
    if not resolved_api_key:
        raise ValueError("Missing API key. Set BFL_API_KEY or pass api_key.")

    image = Path(image_path)
    mask = Path(mask_path)
    validate_image_and_mask(image, mask)

    output = Path(out_path) if out_path else None
    resolved_output_format = output_format or infer_output_format(output)
    if resolved_output_format not in {"jpeg", "png"}:
        raise ValueError("output_format must be 'jpeg' or 'png'.")
    if output is None:
        output = make_default_output_path(resolved_output_format)

    request_id, polling_url = create_request(
        image_path=image,
        mask_path=mask,
        prompt=prompt,
        api_key=resolved_api_key,
        steps=steps,
        guidance=guidance,
        output_format=resolved_output_format,
        safety_tolerance=safety_tolerance,
    )
    if status_callback:
        status_callback(f"Submitted request: {request_id}")

    result = poll_result(
        polling_url=polling_url,
        api_key=resolved_api_key,
        poll_interval=poll_interval,
        timeout=timeout,
        status_callback=status_callback,
    )
    if save_response_json:
        save_json(Path(save_response_json), result)

    download_result(get_result_url(result), output)
    return output


def main() -> None:
    args = parse_args()
    try:
        result_path = generate_inpainted_image(
            image_path=args.image,
            mask_path=args.mask,
            prompt=args.prompt,
            out_path=args.out,
            api_key=args.api_key,
            steps=args.steps,
            guidance=args.guidance,
            output_format=args.output_format,
            safety_tolerance=args.safety_tolerance,
            poll_interval=args.poll_interval,
            timeout=args.timeout,
            save_response_json=args.save_response_json,
            status_callback=lambda status: print(f"Status: {status}"),
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    print(f"Saved result to: {result_path}")


if __name__ == "__main__":
    main()
