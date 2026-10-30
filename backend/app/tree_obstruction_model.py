"""Specialist detector for fallen trees, logs, and timber road obstructions."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from .road_damage_model import Detection

MODEL_DISPLAY_NAME = "Specialist Tree & Obstruction Detector"

TREE_PROMPTS = (
    "fallen tree across roadway",
    "fallen tree across the road",
    "fallen tree",
    "tree trunk on road",
    "fallen log on road",
    "fallen tree blocking traffic",
    "tree branches on road",
)


class TreeObstructionModel:
    """Specialist detector specifically targeting fallen trees and large timber obstructions."""

    def __init__(self):
        import torch
        from transformers.models.auto.configuration_auto import CONFIG_MAPPING
        from transformers import AutoProcessor, GroundingDinoForObjectDetection, BertConfig

        CONFIG_MAPPING.register("grounding-dino-text-prenet", BertConfig)
        self.torch = torch
        self.processor = AutoProcessor.from_pretrained(
            "IDEA-Research/grounding-dino-tiny",
            revision="43032ec967ada6c1831294945b758aa9b7cf1a54",
        )
        self.model = GroundingDinoForObjectDetection.from_pretrained(
            "IDEA-Research/grounding-dino-tiny",
            revision="43032ec967ada6c1831294945b758aa9b7cf1a54",
            use_safetensors=True,
        ).to("cpu").eval()

    def predict_image(self, image: Any, *, frame_second: float = 0.0) -> list[Detection]:
        import numpy as np
        from PIL import Image

        pil_image = image if isinstance(image, Image.Image) else Image.fromarray(np.asarray(image, dtype=np.uint8))
        img_w, img_h = pil_image.width, pil_image.height

        text_prompt = " . ".join(TREE_PROMPTS) + " ."
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

        detections: list[Detection] = []
        for box, score, phrase in zip(results["boxes"], results["scores"], results["labels"]):
            x1, y1, x2, y2 = box
            # Discard full-canvas bounding boxes spanning the entire image
            if (x2 - x1) >= 0.96 * img_w and (y2 - y1) >= 0.90 * img_h:
                continue
            detections.append(Detection(
                frame_second=round(max(0.0, float(frame_second)), 3),
                model_class="fallen_tree",
                confidence=round(float(score), 6),
                bbox_xyxy=tuple(round(float(v), 2) for v in box),
                source_model=f"Specialist Tree Detector ({str(phrase)[:60]})",
                evidence="Fallen tree blocking roadway",
            ))

        # Deduplicate with IoU suppression
        sorted_dets = sorted(detections, key=lambda d: d.confidence, reverse=True)
        keep: list[Detection] = []
        for candidate in sorted_dets:
            cx1, cy1, cx2, cy2 = candidate.bbox_xyxy
            dup = False
            for current in keep:
                kx1, ky1, kx2, ky2 = current.bbox_xyxy
                inter = max(0.0, min(cx2, kx2) - max(cx1, kx1)) * max(0.0, min(cy2, ky2) - max(cy1, ky1))
                union = (cx2 - cx1) * (cy2 - cy1) + (kx2 - kx1) * (ky2 - ky1) - inter
                if union > 0 and inter / union >= 0.40:
                    dup = True
                    break
            if not dup:
                keep.append(candidate)
        return keep

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
