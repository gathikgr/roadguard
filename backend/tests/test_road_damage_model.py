from types import SimpleNamespace
from app.road_damage_model import CLASS_NAMES, RoadDamageModel

class FakeYOLO:
    def predict(self, image, *, conf, iou, imgsz, verbose):
        assert conf == 0.15
        assert iou == 0.45
        assert imgsz in (640, 960)
        assert verbose is False
        box = SimpleNamespace(cls=[0], conf=[0.81234567], xyxy=[[1, 2, 31, 42]])
        return [SimpleNamespace(boxes=[box])]

def test_prediction_returns_unverified_pothole_suggestion():
    model = RoadDamageModel.__new__(RoadDamageModel)
    model.model = FakeYOLO()
    model.confidence_threshold = 0.15
    model.model_path = "checkpoint.pt"
    detections = model.predict_image(object(), frame_second=3.4)
    assert CLASS_NAMES == ("pothole",)
    assert detections[0].to_dict() == {"frame_second":3.4,"model_class":"pothole","confidence":0.812346,"bbox_xyxy":(1.0,2.0,31.0,42.0),"source_model":"peterhdd/pothole-detection-yolov8"}
