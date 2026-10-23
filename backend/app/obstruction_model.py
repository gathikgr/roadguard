"""Grounding DINO fallback focused on fallen trees and road-blocking objects."""
from __future__ import annotations

import os

from .road_damage_model import Detection

MODEL_ID = "IDEA-Research/grounding-dino-tiny"
# Pin model configuration/weights to a reviewed Hugging Face revision.
MODEL_REVISION = "43032ec967ada6c1831294945b758aa9b7cf1a54"
PROMPTS = (
    "fallen tree across the road",
    "fallen tree blocking a road",
    "tree branches blocking traffic",
    "fallen tree lying on a road",
    "debris blocking a road",
)


class ObstructionModel:
    """Open-set detector for tree falls and road obstructions; suggestions only."""

    def __init__(self):
        import torch
        from transformers import AutoProcessor, GroundingDinoForObjectDetection

        self.torch = torch
        self.processor = AutoProcessor.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
        # Use the concrete class: Transformers' generic auto-map can misread
        # the tiny checkpoint's text-prenet entry on recent releases.
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

    def predict_image(self, image, *, frame_second: float = 0.0) -> list[Detection]:
        if self.fallback == "owlvit":
            import numpy as np
            from PIL import Image
            pil_image = image if isinstance(image, Image.Image) else Image.fromarray(np.asarray(image, dtype=np.uint8))
            inputs = self.processor(images=pil_image, text=[list(PROMPTS)], return_tensors="pt")
            with self.torch.inference_mode():
                outputs = self.model(**inputs)
            results = self.processor.post_process_grounded_object_detection(
                outputs=outputs, target_sizes=[(pil_image.height, pil_image.width)], threshold=0.10
            )[0]
            detections = []
            for box, score, label in zip(results["boxes"], results["scores"], results["labels"]):
                phrase = PROMPTS[int(label)] if int(label) < len(PROMPTS) else PROMPTS[0]
                detections.append(Detection(frame_second=round(max(0.0, float(frame_second)), 3), model_class="road_obstruction", confidence=round(float(score), 6), bbox_xyxy=tuple(round(float(value), 2) for value in box), source_model=f"OWL-ViT ({phrase})"))
            return detections
        import numpy as np
        from PIL import Image

        pil_image = image if isinstance(image, Image.Image) else Image.fromarray(np.asarray(image, dtype=np.uint8))
        inputs = self.processor(images=pil_image, text=[list(PROMPTS)], return_tensors="pt")
        with self.torch.inference_mode():
            outputs = self.model(**inputs)
        threshold = float(os.getenv("ROAD_OBSTRUCTION_CONFIDENCE", "0.22"))
        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=threshold,
            text_threshold=max(0.1, threshold - 0.05),
            target_sizes=[(pil_image.height, pil_image.width)],
        )[0]
        detections = []
        for box, score, phrase in zip(results["boxes"], results["scores"], results["labels"]):
            detections.append(Detection(
                frame_second=round(max(0.0, float(frame_second)), 3),
                model_class="road_obstruction",
                confidence=round(float(score), 6),
                bbox_xyxy=tuple(round(float(value), 2) for value in box),
                source_model=f"Grounding DINO Tiny ({str(phrase)[:80]})",
            ))
        return detections

    def scan_video(self, video_path, *, sample_every_seconds: float = 1.0, max_frames: int = 300):
        if self.fallback == "owlvit":
            import cv2
            capture = cv2.VideoCapture(str(video_path)); detections=[]; next_sample=0.0; sampled=0
            try:
                while sampled < max_frames:
                    ok, frame = capture.read()
                    if not ok: break
                    timestamp = capture.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                    if timestamp + 1e-6 < next_sample: continue
                    detections.extend(self.predict_image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), frame_second=timestamp))
                    sampled += 1; next_sample = timestamp + sample_every_seconds
            finally: capture.release()
            return detections
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
