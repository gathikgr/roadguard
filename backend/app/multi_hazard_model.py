"""Optional multi-class model for cracks and waterlogging from the user's repo."""
from __future__ import annotations

import hashlib
import os
import shutil
import urllib.request
from pathlib import Path

from .road_damage_model import Detection

REPO = "yashshree7054/road-hazard-yolov8"
REVISION = "741f91d25d2dfc1cf99b4101cad8b2d4a1cc4613"
WEIGHTS_BLOB_SHA1 = "ee3bedb84ba211e20f5228dcbc438995bd2f381e"
WEIGHTS_URL = f"https://raw.githubusercontent.com/{REPO}/{REVISION}/weights/best.pt"
CLASS_NAMES = {0: "pothole", 1: "crack", 2: "waterlogging"}
CLASS_THRESHOLDS = {
    "pothole": float(os.getenv("ROAD_POTHOLE_CONFIDENCE", "0.15")),
    "crack": float(os.getenv("ROAD_CRACK_CONFIDENCE", "0.06")),
    "waterlogging": float(os.getenv("ROAD_WATERLOGGING_CONFIDENCE", "0.04")),
}


def _download_weights() -> Path:
    cache_root = Path(os.getenv("HF_HOME", Path.home() / ".cache" / "huggingface"))
    target = cache_root / "roadguard" / "road-hazard-yolov8-best.pt"
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_file():
        return target
    partial = target.with_suffix(f".{os.getpid()}.part")
    try:
        request = urllib.request.Request(WEIGHTS_URL, headers={"User-Agent": "RoadGuard/1.0"})
        with urllib.request.urlopen(request, timeout=180) as response, partial.open("wb") as output:
            shutil.copyfileobj(response, output)
        data = partial.read_bytes()
        git_sha = hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()
        if git_sha != WEIGHTS_BLOB_SHA1:
            raise ValueError("Downloaded multi-hazard weights failed the pinned Git object checksum")
        partial.replace(target)
        return target
    except Exception:
        partial.unlink(missing_ok=True)
        raise


class MultiHazardModel:
    """YOLOv8n auxiliary detector. Crack and puddle results are suggestions only."""

    def __init__(self):
        configured = os.getenv("ROAD_MULTI_HAZARD_MODEL_PATH")
        weights = Path(configured).expanduser() if configured else _download_weights()
        if not weights.is_file():
            raise FileNotFoundError(f"Multi-hazard weights not found: {weights}")
        from ultralytics import YOLO
        self.model = YOLO(str(weights))

    def predict_image(self, image, *, frame_second: float = 0.0) -> list[Detection]:
        from .road_damage_model import IMAGE_SIZE, NMS_IOU_THRESHOLD
        if not hasattr(image, "convert"):
            import numpy as np
            from PIL import Image
            image = Image.fromarray(np.asarray(image, dtype=np.uint8)).convert("RGB")
        result = self.model.predict(image, conf=min(CLASS_THRESHOLDS.values()), iou=NMS_IOU_THRESHOLD, imgsz=IMAGE_SIZE, verbose=False)[0]
        detections = []
        for box in result.boxes:
            class_id = int(box.cls[0])
            label = CLASS_NAMES.get(class_id)
            if label is None:
                continue
            confidence = float(box.conf[0])
            if confidence < CLASS_THRESHOLDS[label]:
                continue
            detections.append(Detection(
                frame_second=round(max(0.0, float(frame_second)), 3),
                model_class=label,
                confidence=round(confidence, 6),
                bbox_xyxy=tuple(round(float(value), 2) for value in box.xyxy[0]),
                source_model=f"{REPO}@{REVISION[:7]}",
            ))
        return detections

    def scan_video(self, video_path: str | Path, *, sample_every_seconds: float = 1.0, max_frames: int = 300) -> list[Detection]:
        import cv2
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise ValueError("OpenCV could not decode the supplied video")
        detections: list[Detection] = []
        next_sample, sampled = 0.0, 0
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
