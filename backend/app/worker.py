"""Redis-backed RoadGuard video inference worker."""
from __future__ import annotations
import json
import os
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
import redis
from .road_damage_model import RoadDamageModel
from .multi_hazard_model import MultiHazardModel
from .infrastructure_model import InfrastructureModel
from .obstruction_model import ObstructionModel
from .gemini_model import GeminiVisionModel

REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379/0")
client = redis.Redis.from_url(REDIS_URL, decode_responses=True)


def read_image_metadata(path: str) -> dict:
    """Read safe, useful capture metadata without treating a missing GPS tag as zero."""
    try:
        from PIL import Image
        with Image.open(path) as image:
            exif = image.getexif()
            result: dict = {"captured_at": exif.get(36867) or exif.get(306)}
            gps = exif.get_ifd(34853) if 34853 in exif else {}
            lat, lon = gps.get(2), gps.get(4)
            lat_ref, lon_ref = gps.get(1), gps.get(3)
            if lat and lon and lat_ref and lon_ref:
                def decimal(values):
                    return float(values[0]) + float(values[1]) / 60 + float(values[2]) / 3600
                latitude, longitude = decimal(lat), decimal(lon)
                if str(lat_ref).upper() == "S":
                    latitude = -latitude
                if str(lon_ref).upper() == "W":
                    longitude = -longitude
                if -90 <= latitude <= 90 and -180 <= longitude <= 180:
                    result["location"] = {
                        "latitude": latitude,
                        "longitude": longitude,
                        "source": "IMAGE_EXIF_GPS",
                        "accuracy_m": None,
                    }
            return {key: value for key, value in result.items() if value is not None}
    except Exception:
        # A malformed or metadata-free image should still be analyzed normally.
        return {}


def merge_detections(primary, secondary):
    """Merge overlapping same-class boxes from two models without double-counting."""
    merged = list(primary)
    for candidate in secondary:
        x1, y1, x2, y2 = candidate.bbox_xyxy
        duplicate = None
        for index, current in enumerate(merged):
            if current.model_class != candidate.model_class or current.frame_second != candidate.frame_second:
                continue
            a1, b1, a2, b2 = current.bbox_xyxy
            intersection = max(0, min(x2, a2) - max(x1, a1)) * max(0, min(y2, b2) - max(y1, b1))
            area_a = max(0, x2 - x1) * max(0, y2 - y1)
            area_b = max(0, a2 - a1) * max(0, b2 - b1)
            if intersection / (area_a + area_b - intersection or 1) >= 0.45:
                duplicate = index
                break
        if duplicate is None:
            merged.append(candidate)
        elif candidate.confidence > merged[duplicate].confidence:
            merged[duplicate] = candidate
    return merged


def run() -> None:
    models = {}
    model_errors = {}
    for name, factory in (
        ("pothole", lambda: RoadDamageModel(confidence_threshold=float(os.getenv("ROAD_DAMAGE_CONFIDENCE", "0.15")))),
        ("road_damage_multi_hazard", MultiHazardModel),
        ("infrastructure_open_vocabulary", InfrastructureModel),
        ("grounding_dino_obstruction", ObstructionModel),
        ("gemini_vision_verification", GeminiVisionModel),
    ):
        try:
            models[name] = factory()
        except Exception as exc:
            if name == "grounding_dino_obstruction":
                try:
                    models[name] = ObstructionModel.from_fallback()
                    model_errors[name] = f"Grounding DINO unavailable; using YOLO-World fallback: {str(exc)[:180]}"
                    continue
                except Exception as fallback_exc:
                    model_errors[name] = str(fallback_exc)[:300]
            model_errors[name] = str(exc)[:300]
    while True:
        client.set("roadguard:worker:heartbeat", str(datetime.now(timezone.utc).timestamp()), ex=60)
        client.set("roadguard:worker:models", json.dumps({name: "ready" for name in models} | {name: "unavailable" for name in model_errors if name not in models}), ex=60)
        item = client.blpop("roadguard:video_jobs", timeout=5)
        if item is None:
            continue
        _, raw = item
        payload = json.loads(raw)
        job_id = payload["id"]
        job_key = f"roadguard:job:{job_id}"
        if client.hget(job_key, "status") == "CANCELLED":
            continue
        client.hset(job_key, mapping={"status": "PROCESSING", "started_at": datetime.now(timezone.utc).isoformat()})
        try:
            source_type = payload.get("source_type", "UPLOADED_VIDEO")
            metadata = {}
            if source_type == "UPLOADED_IMAGE":
                import cv2
                metadata = read_image_metadata(payload["path"])
                frame = cv2.imread(payload["path"])
                if frame is None:
                    raise ValueError("OpenCV could not decode the supplied image")
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                detections = []
                for name, detector in models.items():
                    try:
                        detections = merge_detections(detections, detector.predict_image(rgb))
                    except Exception as exc:
                        model_errors[name] = str(exc)[:300]
            else:
                detections = []
                for name, detector in models.items():
                    try:
                        candidates = detector.scan_video(payload["path"], sample_every_seconds=float(os.getenv("VIDEO_SAMPLE_SECONDS", "1")), max_frames=int(os.getenv("VIDEO_MAX_FRAMES", "300")))
                        detections = merge_detections(detections, candidates)
                    except Exception as exc:
                        model_errors[name] = str(exc)[:300]
            model_status = {name: "ready" for name in models} | {name: "unavailable" for name in model_errors if name not in models}
            client.hset(job_key, mapping={"status": "COMPLETED", "completed_at": datetime.now(timezone.utc).isoformat(), "result": json.dumps({"source": source_type, "verification_status": "UNVERIFIED_MODEL_SUGGESTIONS", "model": "Pothole + road damage + open-vocabulary infrastructure pipeline", "models": model_status, "model_errors": model_errors, "metadata": metadata, "location": metadata.get("location"), "detections": [d.to_dict() for d in detections]})})
        except Exception as exc:
            client.hset(job_key, mapping={"status": "FAILED", "completed_at": datetime.now(timezone.utc).isoformat(), "error": str(exc), "traceback": traceback.format_exc()[-4000:]})

if __name__ == "__main__":
    while True:
        try:
            run()
        except Exception:
            time.sleep(3)
