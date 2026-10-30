"""Run road hazard and obstruction model inference on an image or video."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
from .road_damage_model import MODEL_DISPLAY_NAME, MODEL_REPO, RoadDamageModel
from .obstruction_model import ObstructionModel
from .tree_obstruction_model import TreeObstructionModel
from .infrastructure_model import InfrastructureModel
from .multi_hazard_model import MultiHazardModel


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("media", type=Path, help="Image or authorized prerecorded video")
    parser.add_argument("--model", choices=["pothole", "obstruction", "tree", "yolo-world", "multi-hazard", "ensemble"], default="ensemble", help="Model to run")
    parser.add_argument("--weights", type=Path, default=os.getenv("ROAD_DAMAGE_MODEL_PATH"))
    parser.add_argument("--sample-seconds", type=float, default=1.0)
    parser.add_argument("--max-frames", type=int, default=300)
    parser.add_argument("--confidence", type=float, default=None)
    args = parser.parse_args()

    if args.media.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
        try:
            import cv2
        except ImportError as exc:
            raise SystemExit("Install OpenCV with requirements-model.txt") from exc
        frame = cv2.imread(str(args.media))
        if frame is None:
            raise SystemExit("Could not read input image")
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        is_image = True
    else:
        is_image = False

    model_name = args.model
    if model_name == "pothole":
        conf = args.confidence if args.confidence is not None else 0.15
        inst = RoadDamageModel(args.weights, confidence_threshold=conf)
        detections = inst.predict_image(rgb) if is_image else inst.scan_video(args.media, sample_every_seconds=args.sample_seconds, max_frames=args.max_frames)
        display_name = MODEL_DISPLAY_NAME
    elif model_name == "obstruction":
        inst = ObstructionModel()
        detections = inst.predict_image(rgb) if is_image else inst.scan_video(args.media, sample_every_seconds=args.sample_seconds, max_frames=args.max_frames)
        display_name = "Grounding DINO Obstruction Detector"
    elif model_name == "tree":
        inst = TreeObstructionModel()
        detections = inst.predict_image(rgb) if is_image else inst.scan_video(args.media, sample_every_seconds=args.sample_seconds, max_frames=args.max_frames)
        display_name = "Specialist Tree & Obstruction Detector"
    elif model_name == "yolo-world":
        inst = InfrastructureModel()
        detections = inst.predict_image(rgb) if is_image else inst.scan_video(args.media, sample_every_seconds=args.sample_seconds, max_frames=args.max_frames)
        display_name = "Ultralytics YOLO-World Zero-Shot"
    elif model_name == "multi-hazard":
        inst = MultiHazardModel()
        detections = inst.predict_image(rgb) if is_image else inst.scan_video(args.media, sample_every_seconds=args.sample_seconds, max_frames=args.max_frames)
        display_name = "Multi-Hazard YOLOv8"
    else:
        # Full ensemble
        from .worker import load_models, detect_frames
        models, errors = load_models()
        frames = [(rgb, 0.0)] if is_image else []
        if is_image:
            detections = detect_frames(models, frames, errors)
        else:
            from .worker import sample_video
            frames = sample_video(str(args.media), args.sample_seconds, args.max_frames)
            detections = detect_frames(models, frames, errors)
        display_name = "RoadGuard Ensemble (Pothole + Multi-Hazard + YOLO-World + Grounding DINO + Tree Specialist)"

    print(json.dumps({
        "source": "UPLOADED_IMAGE" if is_image else "UPLOADED_VIDEO",
        "verification_status": "UNVERIFIED_MODEL_SUGGESTIONS",
        "model": display_name,
        "detections": [d.to_dict() for d in detections],
    }, indent=2))


if __name__ == "__main__":
    main()
