"""Grounding DINO detector focused on fallen trees and road obstructions."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from .road_damage_model import Detection

MODEL_ID = "IDEA-Research/grounding-dino-tiny"
MODEL_REVISION = "43032ec967ada6c1831294945b758aa9b7cf1a54"

PROMPTS = (
    "fallen tree across the road",
    "fallen tree",
    "tree trunk on road",
    "tree branches blocking road",
    "fallen log",
    "downed power line",
    "traffic cone",
    "road obstruction",
    "debris on road",
)


def _classify_phrase(phrase: str, box: tuple[float, float, float, float], img_h: int, img_w: int) -> tuple[str | None, str | None]:
    p = phrase.lower().strip()
    x1, y1, x2, y2 = box
    # Filter out full-scene / background boxes spanning almost the entire image
    if (x2 - x1) >= 0.96 * img_w and (y2 - y1) >= 0.90 * img_h:
        return None, None
    if any(k in p for k in ("fallen tree", "tree trunk", "tree branches", "fallen log", "log")):
        return "fallen_tree", "Fallen tree blocking roadway"
    if "traffic cone" in p:
        return "road_obstruction", "Traffic cone marking roadway hazard"
    if any(k in p for k in ("power line", "wire", "cable")):
        return "road_obstruction", "Downed power line across road"
    if any(k in p for k in ("obstruction", "debris", "rock", "object")):
        return "road_obstruction", "Road obstruction / debris"
    return None, None


def _nms(detections: list[Detection], iou_thresh: float = 0.40) -> list[Detection]:
    sorted_dets = sorted(detections, key=lambda d: d.confidence, reverse=True)
    keep: list[Detection] = []
    for candidate in sorted_dets:
        x1, y1, x2, y2 = candidate.bbox_xyxy
        duplicate = False
        for current in keep:
            if current.model_class != candidate.model_class or current.frame_second != candidate.frame_second:
                continue
            a1, b1, a2, b2 = current.bbox_xyxy
            intersection = max(0.0, min(x2, a2) - max(x1, a1)) * max(0.0, min(y2, b2) - max(y1, b1))
            area_a = max(0.0, x2 - x1) * max(0.0, y2 - y1)
            area_b = max(0.0, a2 - a1) * max(0.0, b2 - b1)
            union = area_a + area_b - intersection
            if union > 0 and intersection / union >= iou_thresh:
                duplicate = True
                break
        if not duplicate:
            keep.append(candidate)
    return keep


class ObstructionModel:
    """Open-set detector for tree falls and road obstructions."""

    def __init__(self):
        import torch
        from transformers.models.auto.configuration_auto import CONFIG_MAPPING
        from transformers import AutoProcessor, GroundingDinoForObjectDetection, BertConfig

        # Map grounding-dino-text-prenet to BertConfig so Transformers does not raise KeyError
        CONFIG_MAPPING.register("grounding-dino-text-prenet", BertConfig)

        self.torch = torch
        self.processor = AutoProcessor.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
        self.model = GroundingDinoForObjectDetection.from_pretrained(
            MODEL_ID, revision=MODEL_REVISION, use_safetensors=True
        ).to("cpu").eval()
        self.fallback = None

    @classmethod
    def from_fallback(cls):
        """Use OWL-ViT when the DINO checkpoint is incompatible."""
        from transformers import OwlViTProcessor, OwlViTForObjectDetection
        obj = cls.__new__(cls)
        import torch
        obj.torch = torch
        obj.processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
        obj.model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32").to("cpu").eval()
        obj.fallback = "owlvit"
        return obj

    def predict_image(self, image: Any, *, frame_second: float = 0.0) -> list[Detection]:
        import numpy as np
        from PIL import Image

        pil_image = image if isinstance(image, Image.Image) else Image.fromarray(np.asarray(image, dtype=np.uint8))
        img_w, img_h = pil_image.width, pil_image.height

        if self.fallback == "owlvit":
            prompts_list = list(PROMPTS)
            inputs = self.processor(images=pil_image, text=[prompts_list], return_tensors="pt")
            with self.torch.inference_mode():
                outputs = self.model(**inputs)
            threshold = float(os.getenv("ROAD_OBSTRUCTION_CONFIDENCE", "0.04"))
            results = self.processor.post_process_grounded_object_detection(
                outputs=outputs,
                target_sizes=[(img_h, img_w)],
                threshold=threshold,
                text_labels=[prompts_list],
            )[0]
            detections = []
            for box, score, label in zip(results["boxes"], results["scores"], results["text_labels"]):
                cls_name, evidence = _classify_phrase(str(label), tuple(box), img_h, img_w)
                if not cls_name:
                    continue
                detections.append(Detection(
                    frame_second=round(max(0.0, float(frame_second)), 3),
                    model_class=cls_name,
                    confidence=round(float(score), 6),
                    bbox_xyxy=tuple(round(float(v), 2) for v in box),
                    source_model=f"OWL-ViT ({str(label)[:80]})",
                    evidence=evidence,
                ))
            return _nms(detections, iou_thresh=0.40)

        # Grounding DINO expects dot-separated phrases ending with a dot
        text_prompt = " . ".join(PROMPTS) + " ."
        inputs = self.processor(images=pil_image, text=text_prompt, return_tensors="pt")
        with self.torch.inference_mode():
            outputs = self.model(**inputs)

        threshold = float(os.getenv("ROAD_OBSTRUCTION_CONFIDENCE", "0.15"))
        text_threshold = max(0.08, threshold - 0.05)
        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=threshold,
            text_threshold=text_threshold,
            target_sizes=[(img_h, img_w)],
        )[0]

        detections = []
        for box, score, phrase in zip(results["boxes"], results["scores"], results["labels"]):
            cls_name, evidence = _classify_phrase(str(phrase), tuple(box), img_h, img_w)
            if not cls_name:
                continue
            detections.append(Detection(
                frame_second=round(max(0.0, float(frame_second)), 3),
                model_class=cls_name,
                confidence=round(float(score), 6),
                bbox_xyxy=tuple(round(float(value), 2) for value in box),
                source_model=f"Grounding DINO Tiny ({str(phrase)[:80]})",
                evidence=evidence,
            ))
        return _nms(detections, iou_thresh=0.40)

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
