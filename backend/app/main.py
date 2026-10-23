from __future__ import annotations

import math
import logging
import json
import os
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Optional
from uuid import NAMESPACE_URL, uuid4, uuid5

from fastapi import FastAPI, File, Header, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import httpx


class Status(str, Enum):
    DETECTED = "DETECTED"
    PENDING_REVIEW = "PENDING_REVIEW"
    VERIFIED = "VERIFIED"
    PUBLISHED = "PUBLISHED"
    RESOLVED = "RESOLVED"
    REJECTED = "REJECTED"


class Incident(BaseModel):
    id: str
    hazard_type: str
    description: str
    latitude: float
    longitude: float
    severity: str
    confidence: Optional[float] = None
    context_score: Optional[float] = None
    status: Status
    source: str
    location_source: Optional[str] = None
    location_accuracy_m: Optional[float] = None
    evidence_url: Optional[str] = None
    created_at: str
    last_verified_at: Optional[str] = None


class ReportIn(BaseModel):
    description: str = Field(min_length=8, max_length=1000)
    hazard_type: str = Field(min_length=2, max_length=80)
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)


class ReviewIn(BaseModel):
    decision: str
    notes: str = Field(default="", max_length=1000)


class RoutePlanIn(BaseModel):
    origin: str | None = Field(default=None, min_length=2, max_length=200)
    start_latitude: float | None = Field(default=None, ge=-90, le=90)
    start_longitude: float | None = Field(default=None, ge=-180, le=180)
    destination: str = Field(min_length=2, max_length=200)


now = datetime.now(timezone.utc).isoformat()
logger = logging.getLogger("roadguard.events")
incidents: dict[str, Incident] = {
    "demo-01": Incident(id="demo-01", hazard_type="Pothole", description="Demo record: road surface damage near the junction.", latitude=12.9717, longitude=77.5948, severity="HIGH", confidence=0.87, context_score=0.61, status=Status.PUBLISHED, source="DEMO_DATA", created_at=now, last_verified_at=now),
    "demo-02": Incident(id="demo-02", hazard_type="Road debris", description="Demo record: scattered debris in the left lane.", latitude=12.9751, longitude=77.6011, severity="MEDIUM", confidence=0.73, context_score=0.42, status=Status.PENDING_REVIEW, source="UPLOADED_VIDEO", created_at=now),
    "demo-03": Incident(id="demo-03", hazard_type="Cracking", description="Demo record: longitudinal surface cracking.", latitude=12.9674, longitude=77.5883, severity="LOW", confidence=0.68, context_score=0.31, status=Status.REJECTED, source="UPLOADED_VIDEO", created_at=now),
}
audit: list[dict] = []
reports: list[dict] = []
jobs: dict[str, dict] = {}
videos: dict[str, dict] = {}
video_paths: dict[str, str] = {}
published_detection_jobs: set[str] = set()
published_detection_incidents: dict[str, list[str]] = {}
app = FastAPI(title="RoadGuard AI API", version="0.1.0")
origins = [s.strip() for s in os.getenv("API_CORS_ORIGINS", "http://localhost:5173,http://localhost:8080,https://roadguard-ai-gathik.vercel.app,https://roadguard-ai-pi.vercel.app").split(",") if s.strip()]
app.add_middleware(CORSMiddleware, allow_origins=origins, allow_methods=["GET", "POST", "PATCH"], allow_headers=["Content-Type", "X-Admin-Token", "Authorization"])


def admin_guard(token: str | None) -> None:
    """Allow control actions without app login, as requested by the owner."""
    return None


def distance_to_segment_m(point: tuple[float, float], a: tuple[float, float], b: tuple[float, float]) -> float:
    # Equirectangular projection is adequate for a local corridor filter over short distances.
    lat0 = math.radians(point[0])
    scale_x = 111_320 * math.cos(lat0)
    scale_y = 110_540
    px, py = point[1] * scale_x, point[0] * scale_y
    ax, ay = a[1] * scale_x, a[0] * scale_y
    bx, by = b[1] * scale_x, b[0] * scale_y
    dx, dy = bx - ax, by - ay
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy or 1)))
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))


def _supabase_secret() -> str | None:
    return os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_SECRET_KEY")


def _supabase_request(method: str, table: str, *, params: dict | None = None, payload: dict | None = None):
    base_url = os.getenv("SUPABASE_URL", "").rstrip("/")
    secret = _supabase_secret()
    if not base_url or not secret:
        return None
    try:
        response = httpx.request(
            method, f"{base_url}/rest/v1/{table}", params=params, json=payload,
            headers={"apikey": secret, "Authorization": f"Bearer {secret}", "Content-Type": "application/json", "Prefer": "resolution=ignore-duplicates,return=representation" if method == "POST" else "return=representation"},
            timeout=8.0,
        )
        response.raise_for_status()
        return response.json() if response.content else []
    except (httpx.HTTPError, ValueError) as exc:
        logger.warning("Supabase %s on %s failed (%s)", method, table, type(exc).__name__)
        return None


def _sync_supabase_published() -> None:
    """Reload persisted incidents so review data survives API restarts.

    The function name is kept for compatibility with existing callers.  Supabase
    is the source of truth for every review state; callers that serve commuters
    still filter this in-memory collection to PUBLISHED incidents.
    """
    rows = _supabase_request("GET", "incidents", params={"select": "id,hazard_type,description,latitude,longitude,severity,ai_confidence,context_score,location_accuracy_m,status,source_type,created_at,last_verified_at", "limit": "1000"})
    if rows is None:
        return
    for row in rows:
        if row.get("latitude") is None or row.get("longitude") is None:
            continue
        incident_id = str(row["id"])
        incidents[incident_id] = Incident(
            id=incident_id, hazard_type=row["hazard_type"], description=row.get("description", ""),
            latitude=row["latitude"], longitude=row["longitude"], severity=row["severity"],
            confidence=row.get("ai_confidence"), context_score=row.get("context_score"),
            status=Status(row["status"]), source=row["source_type"],
            location_accuracy_m=row.get("location_accuracy_m"), created_at=row["created_at"],
            last_verified_at=row.get("last_verified_at"),
        )


@app.get("/api/health")
def health():
    worker = "not_running"
    model = "configured_not_running"
    models = {}
    try:
        import redis
        queue = redis.Redis.from_url(os.getenv("REDIS_URL", "redis://localhost:6379/0"), decode_responses=True)
        queue.ping()
        heartbeat = queue.get("roadguard:worker:heartbeat")
        raw_models = queue.get("roadguard:worker:models")
        if raw_models:
            models = json.loads(raw_models)
        if heartbeat and datetime.now(timezone.utc).timestamp() - float(heartbeat) < 30:
            worker = "running"
            model = "ready"
    except Exception:
        worker = "not_configured"
        model = "not_configured"
    return {"status": "ok", "database": "supabase" if _supabase_secret() and os.getenv("SUPABASE_URL") else "demo-memory", "video_worker": worker, "model": model, "models": models, "weather": "Open-Meteo client-side", "storage": "local-disk"}


@app.get("/api/incidents")
def list_incidents(status: Optional[Status] = None, x_admin_token: str | None = Header(default=None)):
    admin_guard(x_admin_token)
    _sync_supabase_published()
    rows = list(incidents.values())
    return [x for x in rows if status is None or x.status == status]


@app.get("/api/incidents/{incident_id}")
def get_incident(incident_id: str, x_admin_token: str | None = Header(default=None)):
    incident = incidents.get(incident_id)
    if not incident:
        raise HTTPException(404, "Incident not found")
    if incident.status != Status.PUBLISHED:
        admin_guard(x_admin_token)
    return incident


@app.post("/api/videos", status_code=201)
async def upload_video(file: UploadFile = File(...), x_admin_token: str | None = Header(default=None)):
    admin_guard(x_admin_token)
    allowed = {"video/mp4": ".mp4", "video/quicktime": ".mov", "video/webm": ".webm"}
    extension = allowed.get(file.content_type or "")
    if not extension:
        raise HTTPException(415, "Supported prerecorded video formats: MP4, MOV, WEBM")
    limit = int(os.getenv("MAX_UPLOAD_MB", "200")) * 1024 * 1024
    source_id = str(uuid4())
    root = Path(os.getenv("VIDEO_UPLOAD_DIR", "./uploads")).resolve()
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    path = root / f"{source_id}{extension}"
    size = 0
    try:
        with path.open("xb") as output:
            while chunk := await file.read(1024 * 1024):
                size += len(chunk)
                if size > limit:
                    raise HTTPException(413, "Video exceeds configured upload limit")
                output.write(chunk)
        if size < 16:
            raise HTTPException(400, "Video file is empty or invalid")
        with path.open("rb") as stored:
            signature = stored.read(16)
        valid_signature = (extension in {".mp4", ".mov"} and signature[4:8] == b"ftyp") or (extension == ".webm" and signature[:4] == b"\x1aE\xdf\xa3")
        if not valid_signature:
            raise HTTPException(400, "Video container signature does not match a supported format")
    except Exception:
        path.unlink(missing_ok=True)
        raise
    finally:
        await file.close()
    video = {"id": source_id, "original_name": Path(file.filename or "video").name, "source_type": "UPLOADED_VIDEO", "status": "STORED", "created_at": datetime.now(timezone.utc).isoformat()}
    videos[source_id] = video
    video_paths[source_id] = str(path)
    return video


@app.post("/api/images", status_code=201)
async def upload_image(file: UploadFile = File(...), x_admin_token: str | None = Header(default=None)):
    admin_guard(x_admin_token)
    allowed = {"image/jpeg": ".jpg", "image/png": ".png", "image/webp": ".webp"}
    extension = allowed.get(file.content_type or "")
    if not extension:
        raise HTTPException(415, "Supported road image formats: JPEG, PNG, WEBP")
    limit = int(os.getenv("MAX_UPLOAD_MB", "200")) * 1024 * 1024
    source_id = str(uuid4())
    root = Path(os.getenv("VIDEO_UPLOAD_DIR", "./uploads")).resolve()
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    path = root / f"{source_id}{extension}"
    size = 0
    try:
        with path.open("xb") as output:
            while chunk := await file.read(1024 * 1024):
                size += len(chunk)
                if size > limit:
                    raise HTTPException(413, "Image exceeds configured upload limit")
                output.write(chunk)
        if size < 16:
            raise HTTPException(400, "Image file is empty or invalid")
        with path.open("rb") as stored:
            signature = stored.read(16)
        valid_signature = (
            (extension == ".jpg" and signature[:3] == b"\xff\xd8\xff")
            or (extension == ".png" and signature[:8] == b"\x89PNG\r\n\x1a\n")
            or (extension == ".webp" and signature[:4] == b"RIFF" and signature[8:12] == b"WEBP")
        )
        if not valid_signature:
            raise HTTPException(400, "Image signature does not match a supported image format")
    except Exception:
        path.unlink(missing_ok=True)
        raise
    finally:
        await file.close()
    image = {"id": source_id, "original_name": Path(file.filename or "image").name, "source_type": "UPLOADED_IMAGE", "status": "STORED", "created_at": datetime.now(timezone.utc).isoformat()}
    videos[source_id] = image
    video_paths[source_id] = str(path)
    return image


@app.get("/api/videos")
def list_videos(x_admin_token: str | None = Header(default=None)):
    admin_guard(x_admin_token)
    return list(videos.values())


@app.get("/api/videos/{video_id}")
def get_video(video_id: str, x_admin_token: str | None = Header(default=None)):
    admin_guard(x_admin_token)
    if video_id not in videos:
        raise HTTPException(404, "Video not found")
    return videos[video_id]


@app.post("/api/videos/{video_id}/process", status_code=202)
def process_video(video_id: str, x_admin_token: str | None = Header(default=None)):
    admin_guard(x_admin_token)
    if video_id not in videos:
        raise HTTPException(404, "Video not found")
    try:
        import json
        import redis
        queue = redis.Redis.from_url(os.getenv("REDIS_URL", "redis://localhost:6379/0"), decode_responses=True)
        queue.ping()
    except Exception as exc:
        raise HTTPException(503, "Video worker queue is unavailable; run the configured Redis and worker services") from exc
    job_id = str(uuid4())
    job = {"id": job_id, "video_id": video_id, "status": "QUEUED", "created_at": datetime.now(timezone.utc).isoformat()}
    queue.hset(f"roadguard:job:{job_id}", mapping=job)
    queue.rpush("roadguard:video_jobs", json.dumps({"id": job_id, "video_id": video_id, "path": video_paths[video_id], "source_type": videos[video_id]["source_type"]}))
    return job


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str, x_admin_token: str | None = Header(default=None)):
    admin_guard(x_admin_token)
    try:
        import json
        import redis
        job = redis.Redis.from_url(os.getenv("REDIS_URL", "redis://localhost:6379/0"), decode_responses=True).hgetall(f"roadguard:job:{job_id}")
        if not job:
            raise HTTPException(404, "Job not found")
        if job.get("result"):
            job["result"] = json.loads(job["result"])
            job["published_incident_ids"] = _publish_located_image_detections(job_id, job["result"])
        return job
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(503, "Video worker queue is unavailable") from exc


def _publish_located_image_detections(job_id: str, result: dict) -> list[str]:
    """Place every valid anomaly into the incident review queue.

    EXIF GPS is the camera position, not a surveyed coordinate for each box. The
    incident is therefore deliberately described as an approximate photo location.
    Zero-shot open-vocabulary boxes remain visible suggestions but are withheld from route
    alerts until validated, because they can be semantically broad or wrong. Moving video
    is excluded because its embedded starting GPS cannot locate every frame.
    """
    if job_id in published_detection_jobs:
        return published_detection_incidents.get(job_id, [])
    location = result.get("location") or {}
    latitude, longitude = location.get("latitude"), location.get("longitude")
    detections = result.get("detections") or []
    if not detections:
        return []
    now = datetime.now(timezone.utc).isoformat()
    by_class: dict[str, list[dict]] = {}
    incident_ids: list[str] = []
    for detection in detections:
        source_model = str(detection.get("source_model", ""))
        if source_model.startswith(("Ultralytics YOLO-World", "Grounding DINO")):
            continue
        by_class.setdefault(str(detection.get("model_class", "road damage")), []).append(detection)
    persistence_failed = False
    for hazard_class, class_detections in by_class.items():
        label = hazard_class.replace("_", " ").title()
        confidence = max(float(d.get("confidence", 0)) for d in class_detections)
        incident_id = str(uuid5(NAMESPACE_URL, f"roadguard:{job_id}:{hazard_class}"))
        incident_ids.append(incident_id)
        has_location = latitude is not None and longitude is not None
        incident_latitude, incident_longitude = (float(latitude), float(longitude)) if has_location else (0.0, 0.0)
        incidents[incident_id] = Incident(
            id=incident_id,
            hazard_type=label,
            description=f"AI detected {len(class_detections)} {label.lower()} feature(s). {'Camera GPS is approximate; ' if has_location else 'No GPS was available; ' }verify the evidence before publishing.",
            latitude=incident_latitude, longitude=incident_longitude, severity="MEDIUM",
            confidence=confidence, status=Status.PENDING_REVIEW, source="AI_DETECTION_REVIEW",
            location_source=location.get("source") if has_location else None,
            location_accuracy_m=location.get("accuracy_m"), created_at=now,
            last_verified_at=now,
        )
        persisted = _supabase_request("POST", "incidents", params={"on_conflict": "id"}, payload={
            "id": incident_id, "hazard_type": label, "description": incidents[incident_id].description,
            "latitude": incident_latitude, "longitude": incident_longitude, "severity": "MEDIUM",
            "ai_confidence": confidence, "status": "PENDING_REVIEW", "source_type": "AI_DETECTION_REVIEW",
            "location_accuracy_m": location.get("accuracy_m"), "created_at": now, "last_verified_at": now,
        })
        persistence_failed = persistence_failed or bool(_supabase_secret() and persisted is None)
    if not persistence_failed:
        published_detection_jobs.add(job_id)
        published_detection_incidents[job_id] = incident_ids
    return incident_ids


@app.post("/api/jobs/{job_id}/cancel")
def cancel_job(job_id: str, x_admin_token: str | None = Header(default=None)):
    admin_guard(x_admin_token)
    try:
        import redis
        queue = redis.Redis.from_url(os.getenv("REDIS_URL", "redis://localhost:6379/0"), decode_responses=True)
        key = f"roadguard:job:{job_id}"
        job = queue.hgetall(key)
    except Exception as exc:
        raise HTTPException(503, "Video worker queue is unavailable") from exc
    if not job:
        raise HTTPException(404, "Job not found")
    if job.get("status") != "QUEUED":
        raise HTTPException(409, "Only queued jobs can be cancelled")
    queue.hset(key, mapping={"status": "CANCELLED", "completed_at": datetime.now(timezone.utc).isoformat()})
    job["status"] = "CANCELLED"
    return job


@app.get("/api/hazards/nearby")
def nearby(latitude: float = Query(ge=-90, le=90), longitude: float = Query(ge=-180, le=180), radius_m: int = Query(5000, ge=50, le=50000)):
    _sync_supabase_published()
    rows = [i for i in incidents.values() if i.status == Status.PUBLISHED and distance_to_segment_m((i.latitude, i.longitude), (latitude, longitude), (latitude, longitude)) <= radius_m]
    return {"data_status": "DEMO_DATA" if rows else "NO_VERIFIED_HAZARDS", "hazards": rows}


class RouteQuery(BaseModel):
    geometry: list[tuple[float, float]] = Field(min_length=2, max_length=5000)
    corridor_m: int = Field(default=500, ge=50, le=5000)


def search_place(client, query: str) -> tuple[float, float]:
    response = client.get(
        "https://nominatim.openstreetmap.org/search",
        params={"format": "jsonv2", "limit": 1, "q": query},
    )
    if response.status_code == 429:
        raise HTTPException(429, "Place search is busy. Wait a moment and retry.")
    if response.status_code != 200:
        raise HTTPException(502, "Place search is temporarily unavailable.")
    places = response.json()
    if not places:
        raise HTTPException(404, f"No matching place found for: {query}")
    return float(places[0]["lat"]), float(places[0]["lon"])

@app.get("/api/places/suggest")
def suggest_places(q: str = Query(min_length=2, max_length=120), latitude: float | None = None, longitude: float | None = None):
    """Return nearby place suggestions for destination autocomplete."""
    import httpx
    params = {"format": "jsonv2", "limit": 5, "q": q, "addressdetails": 1}
    if latitude is not None and longitude is not None:
        params.update({"viewbox": f"{longitude-0.35},{latitude+0.35},{longitude+0.35},{latitude-0.35}", "bounded": 0})
    try:
        with httpx.Client(timeout=8.0, headers={"User-Agent": "RoadGuardAI/1.0 (open-source road safety project)"}) as client:
            response = client.get("https://nominatim.openstreetmap.org/search", params=params)
        if response.status_code == 429:
            raise HTTPException(429, "Place search is busy. Wait a moment and retry.")
        response.raise_for_status()
        return {"suggestions": [{"label": p.get("display_name", ""), "latitude": float(p["lat"]), "longitude": float(p["lon"])} for p in response.json()]}
    except HTTPException:
        raise
    except (httpx.HTTPError, ValueError, KeyError) as exc:
        raise HTTPException(502, "Place suggestions are temporarily unavailable.") from exc


@app.post("/api/route/plan")
def plan_route(body: RoutePlanIn):
    import time
    import httpx

    if body.origin:
        if body.start_latitude is not None or body.start_longitude is not None:
            raise HTTPException(422, "Provide a starting place or live GPS coordinates, not both.")
        try:
            with httpx.Client(timeout=8.0, headers={"User-Agent": "RoadGuardAI/1.0 (open-source road safety project)"}) as client:
                start_latitude, start_longitude = search_place(client, body.origin)
                time.sleep(1.05)  # Respect Nominatim's public usage limit.
                end_latitude, end_longitude = search_place(client, body.destination)
        except HTTPException:
            raise
        except httpx.HTTPError as exc:
            raise HTTPException(502, "OpenStreetMap place search failed. Please retry.") from exc
    else:
        if body.start_latitude is None or body.start_longitude is None:
            raise HTTPException(422, "Allow GPS or enter a starting place to plan a route.")
        start_latitude, start_longitude = body.start_latitude, body.start_longitude
        try:
            with httpx.Client(timeout=8.0, headers={"User-Agent": "RoadGuardAI/1.0 (open-source road safety project)"}) as client:
                end_latitude, end_longitude = search_place(client, body.destination)
        except HTTPException:
            raise
        except httpx.HTTPError as exc:
            raise HTTPException(502, "OpenStreetMap place search failed. Please retry.") from exc

    try:
        response = httpx.get(
            f"https://router.project-osrm.org/route/v1/driving/{start_longitude},{start_latitude};{end_longitude},{end_latitude}",
            params={"overview": "full", "geometries": "geojson", "steps": "false"},
            timeout=12.0,
            headers={"User-Agent": "RoadGuardAI/1.0 (open-source road safety project)"},
        )
        response.raise_for_status()
        result = response.json()
    except httpx.HTTPError as exc:
        raise HTTPException(502, "OpenStreetMap route service is temporarily unavailable.") from exc
    if result.get("code") != "Ok" or not result.get("routes"):
        raise HTTPException(404, "A driving route could not be found between these places.")
    route = result["routes"][0]
    coordinates = route["geometry"]["coordinates"]
    stride = max(1, math.ceil(len(coordinates) / 5000))
    points = [{"latitude": lat, "longitude": lon} for lon, lat in coordinates[::stride]]
    if coordinates and (coordinates[-1][0], coordinates[-1][1]) != (coordinates[::stride][-1][0], coordinates[::stride][-1][1]):
        points.append({"latitude": coordinates[-1][1], "longitude": coordinates[-1][0]})
    return {
        "geometry": points,
        "distance_m": route["distance"],
        "duration_s": route["duration"],
        "start": {"latitude": start_latitude, "longitude": start_longitude},
        "destination": {"latitude": end_latitude, "longitude": end_longitude},
        "directions_url": f"https://www.openstreetmap.org/directions?engine=fossgis_osrm_car&route={start_latitude}%2C{start_longitude}%3B{end_latitude}%2C{end_longitude}",
    }


@app.post("/api/hazards/route-query")
def route_query(body: RouteQuery):
    _sync_supabase_published()
    active = [i for i in incidents.values() if i.status == Status.PUBLISHED]
    found = []
    for incident in active:
        distance = min(distance_to_segment_m((incident.latitude, incident.longitude), body.geometry[j], body.geometry[j + 1]) for j in range(len(body.geometry) - 1))
        if distance <= body.corridor_m:
            found.append({**incident.model_dump(), "distance_to_route_m": round(distance)})
    return {"data_status": "DEMO_DATA" if found else "NO_VERIFIED_HAZARDS", "hazards": found}


@app.post("/api/reports", status_code=201)
def submit_report(body: ReportIn):
    report = {"id": str(uuid4()), **body.model_dump(), "status": "PENDING_REVIEW", "source": "USER_REPORT", "created_at": datetime.now(timezone.utc).isoformat()}
    reports.append(report)
    return report


@app.get("/api/reports")
def list_reports(x_admin_token: str | None = Header(default=None)):
    admin_guard(x_admin_token)
    return reports


@app.patch("/api/incidents/{incident_id}/review")
def review(incident_id: str, body: ReviewIn, x_admin_token: str | None = Header(default=None)):
    admin_guard(x_admin_token)
    incident = incidents.get(incident_id)
    if not incident:
        raise HTTPException(404, "Incident not found")
    transitions = {Status.DETECTED: {Status.PENDING_REVIEW}, Status.PENDING_REVIEW: {Status.VERIFIED, Status.REJECTED}, Status.VERIFIED: {Status.PUBLISHED, Status.REJECTED}, Status.PUBLISHED: {Status.RESOLVED}}
    decision = body.decision.upper()
    if decision not in {x.value for x in transitions.get(incident.status, set())}:
        raise HTTPException(409, f"Invalid transition from {incident.status.value} to {decision}")
    previous = incident.status
    incident.status = Status(decision)
    if incident.status in {Status.VERIFIED, Status.PUBLISHED}:
        incident.last_verified_at = datetime.now(timezone.utc).isoformat()
    audit.append({"incident_id": incident_id, "action": decision, "actor": "local-admin", "from": previous.value, "to": decision, "notes": body.notes, "at": datetime.now(timezone.utc).isoformat()})
    return incident


@app.get("/api/analytics/summary")
def analytics():
    return {"source": "DEMO_DATA", "incidents_total": len(incidents), "published": sum(i.status == Status.PUBLISHED for i in incidents.values()), "pending_review": sum(i.status == Status.PENDING_REVIEW for i in incidents.values()), "reports_pending": len(reports), "audit_events": len(audit)}


@app.get("/api/integrations/status")
def integrations():
    queue_ready = False
    worker_ready = False
    try:
        import redis
        queue = redis.Redis.from_url(os.getenv("REDIS_URL", "redis://localhost:6379/0"), decode_responses=True)
        queue_ready = bool(queue.ping())
        heartbeat = queue.get("roadguard:worker:heartbeat")
        worker_ready = bool(heartbeat and datetime.now(timezone.utc).timestamp() - float(heartbeat) < 30)
    except Exception:
        pass
    return {"supabase": "configured" if os.getenv("SUPABASE_URL") else "not_configured", "database": "supabase" if _supabase_secret() and os.getenv("SUPABASE_URL") else "demo-memory", "weather": "client_side_open_meteo", "video_worker": "running" if worker_ready else ("queued_worker_not_running" if queue_ready else "not_configured"), "model": "ready" if worker_ready else "configured_not_running"}
