"""Pothole-only YOLOv8s inference for uploaded road images and videos."""
from __future__ import annotations

import os
import shutil
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

MODEL_REPO = "peterhdd/pothole-detection-yolov8"
MODEL_FILE = "best.pt"
MODEL_URL = f"https://huggingface.co/{MODEL_REPO}/resolve/main/{MODEL_FILE}"
MODEL_DISPLAY_NAME = "YOLOv8s · pothole-only"
CLASS_NAMES = ("pothole",)
IMAGE_SIZE = 640
NMS_IOU_THRESHOLD = 0.45


@dataclass(frozen=True)
class Detection:
    frame_second: float
    model_class: str
    confidence: float
    bbox_xyxy: tuple[float, float, float, float]
    source_model: str | None = None
    evidence: str | None = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        if data["source_model"] is None:
            data.pop("source_model")
        if data["evidence"] is None:
            data.pop("evidence")
        return data


class RoadDamageModel:
    """Load an Apache-2.0 YOLOv8s pothole checkpoint and infer locally."""

    def __init__(self, model_path: str | Path | None = None, *, confidence_threshold: float = 0.15):
        if not 0 <= confidence_threshold <= 1:
            raise ValueError("confidence_threshold must be between 0 and 1")

        configured_path = model_path or os.getenv("ROAD_DAMAGE_MODEL_PATH")
        if configured_path:
            weights = Path(configured_path).expanduser().resolve()
            if not weights.is_file():
                raise FileNotFoundError(f"Road-damage weights not found: {weights}")
            weights = str(weights)
        else:
            cache_root = Path(os.getenv("HF_HOME", Path.home() / ".cache" / "huggingface"))
            cached_weights = cache_root / "roadguard" / MODEL_FILE
            cached_weights.parent.mkdir(parents=True, exist_ok=True)
            if not cached_weights.is_file():
                partial_weights = cached_weights.with_name(f"{MODEL_FILE}.{os.getpid()}.part")
                request = urllib.request.Request(MODEL_URL, headers={"User-Agent": "RoadGuard/1.0"})
                try:
                    with urllib.request.urlopen(request, timeout=180) as response, partial_weights.open("wb") as output:
                        shutil.copyfileobj(response, output)
                    partial_weights.replace(cached_weights)
                except Exception:
                    partial_weights.unlink(missing_ok=True)
                    raise
            weights = str(cached_weights)

        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError("Install worker dependencies with pip install -r requirements-model.txt") from exc

        self.model = YOLO(weights)
        self.confidence_threshold = confidence_threshold
        self.model_path = weights

    def predict_image(self, image: Any, *, frame_second: float = 0.0) -> list[Detection]:
        """Accept an RGB PIL image/array and return pixel-space pothole boxes."""
        try:
            import numpy as np
            from PIL import Image
        except ImportError as exc:
            raise RuntimeError("Pillow and NumPy are required for image inference") from exc

        # The worker decodes frames as RGB. Pass a PIL RGB image so Ultralytics
        # does not reinterpret the array as BGR and shift colors on wet roads.
        source = Image.fromarray(np.asarray(image, dtype=np.uint8)) if isinstance(image, np.ndarray) else image
        result = self.model.predict(
            source,
            conf=self.confidence_threshold,
            iou=NMS_IOU_THRESHOLD,
            imgsz=IMAGE_SIZE,
            verbose=False,
        )[0]

        detections = []
        for box in result.boxes:
            class_id = int(box.cls[0])
            # This checkpoint only has one class. Do not relabel any unexpected
            # class from a user-supplied checkpoint as a pothole.
            if class_id != 0:
                continue
            detections.append(Detection(
                frame_second=round(max(0.0, float(frame_second)), 3),
                model_class=CLASS_NAMES[0],
                confidence=round(float(box.conf[0]), 6),
                bbox_xyxy=tuple(round(float(value), 2) for value in box.xyxy[0]),
                source_model="peterhdd/pothole-detection-yolov8",
            ))
        return detections

    def scan_video(
        self,
        video_path: str | Path,
        *,
        sample_every_seconds: float = 1.0,
        max_frames: int = 300,
    ) -> list[Detection]:
        if sample_every_seconds <= 0 or max_frames < 1:
            raise ValueError("sample_every_seconds and max_frames must be positive")
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError("OpenCV is required for video decoding") from exc
        path = Path(video_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Video not found: {path}")
        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            raise ValueError("OpenCV could not decode the supplied video")
        detections: list[Detection] = []
        next_sample = 0.0
        sampled = 0
        try:
            while sampled < max_frames:
                ok, frame = capture.read()
                if not ok:
                    break
                timestamp = capture.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                if timestamp + 1e-6 < next_sample:
                    continue
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                detections.extend(self.predict_image(rgb, frame_second=timestamp))
                sampled += 1
                next_sample = timestamp + sample_every_seconds
        finally:
            capture.release()
        return detections
