"""Run pothole YOLOv8 inference on an image or prerecorded video."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
from .road_damage_model import MODEL_DISPLAY_NAME, MODEL_REPO, RoadDamageModel


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("media", type=Path, help="Image or authorized prerecorded video")
    parser.add_argument("--weights", type=Path, default=os.getenv("ROAD_DAMAGE_MODEL_PATH"))
    parser.add_argument("--sample-seconds", type=float, default=1.0)
    parser.add_argument("--max-frames", type=int, default=300)
    parser.add_argument("--confidence", type=float, default=0.15)
    args = parser.parse_args()
    model = RoadDamageModel(args.weights, confidence_threshold=args.confidence)
    if args.media.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
        try:
            import cv2
        except ImportError as exc:
            raise SystemExit("Install OpenCV with requirements-model.txt") from exc
        frame = cv2.imread(str(args.media))
        if frame is None:
            raise SystemExit("Could not read input image")
        detections = model.predict_image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    else:
        detections = model.scan_video(args.media, sample_every_seconds=args.sample_seconds, max_frames=args.max_frames)
    print(json.dumps({"source": "UPLOADED_MEDIA", "verification_status": "UNVERIFIED_MODEL_SUGGESTIONS", "model": MODEL_DISPLAY_NAME, "model_repository": MODEL_REPO, "weights": model.model_path, "detections": [d.to_dict() for d in detections]}, indent=2))

if __name__ == "__main__":
    main()
