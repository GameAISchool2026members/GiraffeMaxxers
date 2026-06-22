# yolo_segmentation.py

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


PERSON_CLASS_ID = 0
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "yolo26n-seg.pt"


@dataclass
class YoloSegmentationConfig:
    model_path: str = str(DEFAULT_MODEL_PATH)
    device: str = "cpu"
    conf: float = 0.25
    iou: float = 0.7
    imgsz: int = 960
    mask_threshold: float = 0.5
    choose: str = "largest"  # "largest" or "highest_conf"


class YoloPersonSegmenter:
    def __init__(self, config: YoloSegmentationConfig | None = None):
        self.config = config or YoloSegmentationConfig()
        self.model = YOLO(self.config.model_path)

    def segment_frame(self, frame_bgr: np.ndarray) -> np.ndarray | None:
        result = self.model(
            frame_bgr,
            device=self.config.device,
            conf=self.config.conf,
            iou=self.config.iou,
            imgsz=self.config.imgsz,
            classes=[PERSON_CLASS_ID],
            verbose=False,
        )[0]

        if result.masks is None or result.boxes is None or len(result.boxes) == 0:
            return None

        candidates = []

        for i, box in enumerate(result.boxes):
            cls = int(box.cls[0])
            if cls != PERSON_CLASS_ID:
                continue

            conf = float(box.conf[0])
            raw_mask = result.masks.data[i].cpu().numpy()

            mask = cv2.resize(
                raw_mask,
                (frame_bgr.shape[1], frame_bgr.shape[0]),
                interpolation=cv2.INTER_LINEAR,
            )

            area = int(np.count_nonzero(mask > self.config.mask_threshold))
            candidates.append({
                "index": i,
                "conf": conf,
                "area": area,
                "mask": mask,
            })

        if not candidates:
            return None

        if self.config.choose == "highest_conf":
            best = max(candidates, key=lambda item: item["conf"])
        else:
            best = max(candidates, key=lambda item: item["area"])

        return best["mask"] > self.config.mask_threshold

    def segment_jpeg_to_mask_png(self, jpeg_bytes: bytes) -> bytes:
        frame = decode_image(jpeg_bytes)
        mask = self.segment_frame(frame)

        if mask is None:
            mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        else:
            mask = mask.astype(np.uint8) * 255

        return encode_png(mask)


def decode_image(image_bytes: bytes) -> np.ndarray:
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    if frame is None:
        raise ValueError("Could not decode image bytes")

    return frame


def encode_png(image: np.ndarray) -> bytes:
    ok, encoded = cv2.imencode(".png", image)

    if not ok:
        raise RuntimeError("Could not encode PNG")

    return encoded.tobytes()


# Optional singleton for server use.
_segmenter: YoloPersonSegmenter | None = None


def get_segmenter() -> YoloPersonSegmenter:
    global _segmenter

    if _segmenter is None:
        config = YoloSegmentationConfig(
            model_path=str(DEFAULT_MODEL_PATH),
            device="cpu",
            conf=0.25,
            iou=0.7,
            imgsz=960,
            mask_threshold=0.5,
            choose="largest",
        )
        _segmenter = YoloPersonSegmenter(config)

    return _segmenter


def segment_jpeg_to_mask_png(jpeg_bytes: bytes) -> bytes:
    return get_segmenter().segment_jpeg_to_mask_png(jpeg_bytes)
