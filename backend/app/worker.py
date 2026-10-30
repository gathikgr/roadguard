"""Redis-backed RoadGuard worker for the local YOLOv8 detection ensemble."""
from __future__ import annotations

import json
import os
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import redis

from .road_damage_model import Detection, RoadDamageModel
from .multi_hazard_model import MultiHazardModel
from .infrastructure_model import InfrastructureModel

REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379/0")
client = redis.Redis.from_url(REDIS_URL, decode_responses=True)


def read_image_metadata(path: str) -> dict:
    """Read safe capture metadata without treating missing GPS as zero."""
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
                    result["location"] = {"latitude": latitude, "longitude": longitude, "source": "IMAGE_EXIF_GPS", "accuracy_m": None}
            return {key: value for key, value in result.items() if value is not None}
    except Exception:
        return {}


def merge_detections(primary: list[Detection], secondary: list[Detection]) -> list[Detection]:
    """Merge same-frame, same-class overlaps while retaining the stronger box."""
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


def sample_video(path: str, every_seconds: float, max_frames: int):
    """Decode at most the configured number of frames and reuse them for every model."""
    import cv2

    capture = cv2.VideoCapture(str(Path(path).resolve()))
    if not capture.isOpened():
        raise ValueError("OpenCV could not decode the supplied video")
    fps = capture.get(cv2.CAP_PROP_FPS) or 0.0
    frame_count = capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
    duration = frame_count / fps if fps > 0 else 0.0
    interval = max(every_seconds, duration / max_frames) if duration > 0 else every_seconds
    frames = []
    requested_time = 0.0
    try:
        while len(frames) < max_frames and (duration <= 0 or requested_time < duration):
            capture.set(cv2.CAP_PROP_POS_MSEC, requested_time * 1000)
            ok, frame = capture.read()
            if not ok:
                break
            actual_time = capture.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            timestamp = round(actual_time if actual_time > 0 else requested_time, 3)
            frames.append((rgb, timestamp))
            requested_time += interval
    finally:
        capture.release()
    if not frames:
        raise ValueError("No readable frames were found in the supplied video")
    return frames


def load_models():
    """Load available models independently so one missing checkpoint does not disable the rest."""
    models = {}
    errors = {}
    factories = (
        ("pothole", lambda: RoadDamageModel(confidence_threshold=float(os.getenv("ROAD_DAMAGE_CONFIDENCE", "0.15")))),
        ("road_damage_multi_hazard", MultiHazardModel),
        ("infrastructure_open_vocabulary", InfrastructureModel),
    )
    for name, factory in factories:
        try:
            models[name] = factory()
        except Exception as exc:
            errors[name] = str(exc)[:300]
    return models, errors


def get_model_status(models, errors):
    status = {}
    for name in models:
        error = errors.get(name, "")
        status[name] = "ready_fallback" if "using OWL-ViT" in error else ("error" if error else "ready")
    status.update({name: "unavailable" for name in errors if name not in models})
    return status


def detect_frames(models, frames, errors):
    proposals: list[Detection] = []
    for image, timestamp in frames:
        for name, detector in models.items():
            try:
                predictions = detector.predict_image(image, frame_second=timestamp)
                proposals = merge_detections(proposals, predictions)
            except Exception as exc:
                errors[name] = str(exc)[:300]
    return proposals


def run() -> None:
    models, model_errors = load_models()

    while True:
        status = get_model_status(models, model_errors)
        client.set("roadguard:worker:heartbeat", str(datetime.now(timezone.utc).timestamp()), ex=60)
        client.set("roadguard:worker:models", json.dumps(status), ex=60)
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
            from .instant_detections import lookup_media
            instant = lookup_media(payload["path"], source_type=source_type, client=client)
            if instant is not None:
                client.hset(job_key, mapping={"status": "COMPLETED", "completed_at": datetime.now(timezone.utc).isoformat(), "result": json.dumps(instant)})
                continue

            metadata = {}
            if source_type == "UPLOADED_IMAGE":
                from PIL import Image
                metadata = read_image_metadata(payload["path"])
                with Image.open(payload["path"]) as image:
                    frames = [(image.convert("RGB"), 0.0)]
            else:
                frames = sample_video(
                    payload["path"],
                    every_seconds=float(os.getenv("VIDEO_SAMPLE_SECONDS", "2")),
                    max_frames=int(os.getenv("VIDEO_MAX_FRAMES", "12")),
                )

            proposals = detect_frames(models, frames, model_errors)
            model_status = get_model_status(models, model_errors)
            result = {
                "source": source_type,
                "verification_status": "UNVERIFIED_MODEL_SUGGESTIONS",
                "model": "Pothole + road damage + open-vocabulary infrastructure pipeline",
                "models": model_status,
                "model_errors": model_errors,
                "metadata": metadata,
                "location": metadata.get("location"),
                "detections": [d.to_dict() for d in proposals],
            }
            client.hset(job_key, mapping={"status": "COMPLETED", "completed_at": datetime.now(timezone.utc).isoformat(), "result": json.dumps(result)})
            try:
                import hashlib
                sha = hashlib.sha256(Path(payload["path"]).read_bytes()).hexdigest()
                client.set(f"roadguard:cache:{sha}", json.dumps(result), ex=86400 * 30)
            except Exception:
                pass
        except Exception as exc:
            client.hset(job_key, mapping={"status": "FAILED", "completed_at": datetime.now(timezone.utc).isoformat(), "error": str(exc), "traceback": traceback.format_exc()[-4000:]})


if __name__ == "__main__":
    while True:
        try:
            run()
        except Exception:
            time.sleep(3)
