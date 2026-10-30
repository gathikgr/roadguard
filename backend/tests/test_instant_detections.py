from pathlib import Path
from app.instant_detections import lookup_image, lookup_media, KNOWN_PROFILES


def test_instant_detections_has_three_demo_profiles():
    assert len(KNOWN_PROFILES) >= 3
    names = {p["name"] for p in KNOWN_PROFILES}
    assert "fallen_tree" in names
    assert "pothole_crater" in names
    assert "waterlogging_atv" in names


def test_fallen_tree_detections_contain_required_classes():
    profile = next(p for p in KNOWN_PROFILES if p["name"] == "fallen_tree")
    classes = {d["model_class"] for d in profile["detections"]}
    assert "fallen_tree" in classes
    assert "road_obstruction" in classes


def test_pothole_detections_contain_required_classes():
    profile = next(p for p in KNOWN_PROFILES if p["name"] == "pothole_crater")
    classes = {d["model_class"] for d in profile["detections"]}
    assert "pothole" in classes
    assert "crack" in classes


def test_waterlogging_detections_contain_required_classes():
    profile = next(p for p in KNOWN_PROFILES if p["name"] == "waterlogging_atv")
    classes = {d["model_class"] for d in profile["detections"]}
    assert "waterlogging" in classes
    assert "road_obstruction" in classes
