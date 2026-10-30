from fastapi.testclient import TestClient

from app.main import app, videos

client = TestClient(app)
ADMIN = {"X-Admin-Token": "local-admin"}


def test_public_hazards_exclude_pending_rejected_and_resolved():
    response = client.get("/api/hazards/nearby", params={"latitude": 12.9717, "longitude": 77.5948, "radius_m": 50000})
    assert response.status_code == 200
    assert {row["status"] for row in response.json()["hazards"]} == {"PUBLISHED"}


def test_incident_records_accessible_without_login():
    assert client.get("/api/incidents").status_code == 200
    assert client.get("/api/incidents/demo-01").status_code == 200
    assert client.get("/api/incidents", headers=ADMIN).status_code == 200


def test_route_query_uses_route_corridor_and_published_only():
    response = client.post("/api/hazards/route-query", json={"geometry": [[12.9717, 77.5948], [12.9718, 77.5949]], "corridor_m": 100})
    assert response.status_code == 200
    assert all(row["status"] == "PUBLISHED" for row in response.json()["hazards"])
    assert all(row["distance_to_route_m"] <= 100 for row in response.json()["hazards"])


def test_report_is_pending_and_location_explicitly_supplied():
    response = client.post("/api/reports", json={"hazard_type": "Pothole", "description": "Deep pothole in left lane", "latitude": 12.9, "longitude": 77.5})
    assert response.status_code == 201
    assert response.json()["status"] == "PENDING_REVIEW"
    assert response.json()["source"] == "USER_REPORT"


def test_review_rejects_invalid_transition():
    assert client.patch("/api/incidents/demo-02/review", json={"decision": "PUBLISHED"}).status_code == 409


def test_model_missing_does_not_create_fake_processing_job(monkeypatch):
    monkeypatch.delenv("ROAD_DAMAGE_MODEL_PATH", raising=False)
    videos["test-video"] = {"id": "test-video"}
    response = client.post("/api/videos/test-video/process", headers=ADMIN)
    assert response.status_code == 503


def test_upload_enforces_supported_container_and_writes_private_file(monkeypatch, tmp_path):
    monkeypatch.setenv("VIDEO_UPLOAD_DIR", str(tmp_path))
    good_mp4_header = b"\x00\x00\x00\x18ftypisom" + b"\x00\x00\x00\x00"
    response = client.post("/api/videos", headers=ADMIN, files={"file": ("road.mp4", good_mp4_header, "video/mp4")})
    assert response.status_code == 201
    assert len(list(tmp_path.iterdir())) == 1
    bad = client.post("/api/videos", headers=ADMIN, files={"file": ("fake.mp4", b"not really a video at all", "video/mp4")})
    assert bad.status_code == 400
    assert len(list(tmp_path.iterdir())) == 1

