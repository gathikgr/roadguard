"""Open-vocabulary suggestions for road obstructions and damaged assets.

YOLO-World is a zero-shot prompt detector. Its results are suggestions only;
it cannot reliably establish that a sign or signal is damaged or unsafe.
"""
from __future__ import annotations

import os
from pathlib import Path

from .road_damage_model import Detection, IMAGE_SIZE, NMS_IOU_THRESHOLD

MODEL_FILE = "yolov8s-worldv2.pt"
MODEL_URL = f"https://github.com/ultralytics/assets/releases/download/v8.3.0/{MODEL_FILE}"
PROMPTS = (
    ("standing water on road", "waterlogging"),
    ("flooded road", "waterlogging"),
    ("puddle on road", "waterlogging"),
    ("deep water on road", "waterlogging"),
    ("waterlogging on road", "waterlogging"),
    ("pothole", "pothole"),
    ("road pothole", "pothole"),
    ("debris on road", "road_obstruction"),
    ("fallen tree", "fallen_tree"),
    ("tree trunk on road", "fallen_tree"),
    ("fallen log", "fallen_tree"),
    ("fallen tree blocking road", "fallen_tree"),
    ("log on road", "fallen_tree"),
    ("tree branch on road", "fallen_tree"),
    ("traffic cone", "road_obstruction"),
    ("downed power line", "road_obstruction"),
    ("rock blocking road", "road_obstruction"),
    ("large object blocking road", "road_obstruction"),
    ("damaged traffic sign", "damaged_traffic_sign"),
    ("bent traffic sign", "damaged_traffic_sign"),
    ("damaged traffic light", "damaged_traffic_signal"),
    ("broken traffic light", "damaged_traffic_signal"),
    ("damaged guardrail", "damaged_guardrail"),
    ("open manhole on road", "open_manhole"),
    ("damaged street light", "damaged_streetlight"),
)
PROMPT_NAMES = [prompt for prompt, _ in PROMPTS]
PROMPT_TO_CLASS = {prompt: label for prompt, label in PROMPTS}


class InfrastructureModel:
    """YOLO-World zero-shot detector for likely obstructions and asset damage."""

    def __init__(self):
        from ultralytics import YOLO

        configured = os.getenv("ROAD_INFRASTRUCTURE_MODEL_PATH")
        if configured:
            weights = Path(configured).expanduser()
        elif Path(f"/app/weights/{MODEL_FILE}").is_file():
            weights = Path(f"/app/weights/{MODEL_FILE}")
        else:
            weights = MODEL_FILE
        self.model = YOLO(str(weights))
        self.model.set_classes(PROMPT_NAMES)

    def predict_image(self, image, *, frame_second: float = 0.0) -> list[Detection]:
        if not hasattr(image, "convert"):
            import numpy as np
            from PIL import Image
            image = Image.fromarray(np.asarray(image, dtype=np.uint8)).convert("RGB")
        result = self.model.predict(
            image,
            conf=float(os.getenv("ROAD_INFRASTRUCTURE_CONFIDENCE", "0.035")),
            iou=NMS_IOU_THRESHOLD,
            imgsz=640,
            verbose=False,
        )[0]
        names = result.names
        detections = []
        for box in result.boxes:
            prompt = names[int(box.cls[0])]
            label = PROMPT_TO_CLASS.get(str(prompt))
            if label is None:
                continue
            detections.append(Detection(
                frame_second=round(max(0.0, float(frame_second)), 3),
                model_class=label,
                confidence=round(float(box.conf[0]), 6),
                bbox_xyxy=tuple(round(float(value), 2) for value in box.xyxy[0]),
                source_model="Ultralytics YOLO-World zero-shot",
            ))
        return detections

    def scan_video(self, video_path, *, sample_every_seconds: float = 1.0, max_frames: int = 300):
        import cv2

        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise ValueError("OpenCV could not decode the supplied video")
        detections = []
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
