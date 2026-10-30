from types import SimpleNamespace
from app.obstruction_model import _classify_phrase, _nms
from app.road_damage_model import Detection


def test_classify_phrase_detects_fallen_tree():
    cls, evidence = _classify_phrase("fallen tree across the road", (10.0, 20.0, 200.0, 100.0), 450, 675)
    assert cls == "fallen_tree"
    assert "Fallen tree" in evidence


def test_classify_phrase_detects_tree_trunk_and_branches():
    cls1, _ = _classify_phrase("tree trunk on road", (10.0, 20.0, 200.0, 100.0), 450, 675)
    assert cls1 == "fallen_tree"

    cls2, _ = _classify_phrase("tree branches blocking road", (10.0, 20.0, 200.0, 100.0), 450, 675)
    assert cls2 == "fallen_tree"


def test_classify_phrase_detects_traffic_cones_and_power_lines():
    cls_cone, _ = _classify_phrase("traffic cone", (300.0, 300.0, 400.0, 400.0), 450, 675)
    assert cls_cone == "road_obstruction"

    cls_wire, _ = _classify_phrase("downed power line", (50.0, 50.0, 300.0, 100.0), 450, 675)
    assert cls_wire == "road_obstruction"


def test_classify_phrase_ignores_full_canvas_background():
    # Spans full canvas
    cls, _ = _classify_phrase("tree road fallen tree blocking road", (0.0, 0.0, 670.0, 440.0), 450, 675)
    assert cls is None


def test_nms_suppresses_overlapping_boxes():
    d1 = Detection(frame_second=0.0, model_class="fallen_tree", confidence=0.80, bbox_xyxy=(10.0, 10.0, 100.0, 100.0))
    d2 = Detection(frame_second=0.0, model_class="fallen_tree", confidence=0.60, bbox_xyxy=(12.0, 12.0, 98.0, 98.0))
    d3 = Detection(frame_second=0.0, model_class="fallen_tree", confidence=0.75, bbox_xyxy=(300.0, 300.0, 400.0, 400.0))
    kept = _nms([d1, d2, d3], iou_thresh=0.40)
    assert len(kept) == 2
    assert kept[0].confidence == 0.80
    assert kept[1].confidence == 0.75
