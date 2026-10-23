"""Optional Gemini visual verification stage for difficult obstructions."""
from __future__ import annotations
import base64, json, os, re
import httpx
from .road_damage_model import Detection

PROMPT = """Inspect this road image for real physical hazards. Return ONLY JSON: {\"detections\":[{\"class\":\"fallen_tree|pothole|waterlogging|road_obstruction|other\",\"confidence\":0.0,\"box\":[x1,y1,x2,y2],\"reason\":\"short evidence\"}]}. Coordinates must be pixel coordinates in the original image. Include a fallen tree only if visible across/on the road. Do not infer objects outside the image or invent hazards."""

class GeminiVisionModel:
    def __init__(self):
        self.key = os.getenv("GEMINI_API_KEY", "").strip()
        if not self.key:
            raise RuntimeError("GEMINI_API_KEY is not configured")
        configured = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        self.model = "gemini-3-flash-preview" if configured in {"gemini-2.0-flash", "gemini-2.0-flash-001", "gemini-2.5-flash"} else configured

    def predict_image(self, image, *, frame_second: float = 0.0):
        import cv2, numpy as np
        ok, encoded = cv2.imencode(".jpg", np.asarray(image, dtype=np.uint8))
        if not ok: raise ValueError("Could not encode image for Gemini")
        body = {"contents":[{"parts":[{"text":PROMPT},{"inline_data":{"mime_type":"image/jpeg","data":base64.b64encode(encoded.tobytes()).decode()}}]}], "generationConfig":{"temperature":0,"responseMimeType":"application/json"}}
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent?key={self.key}"
        response = httpx.post(url, json=body, timeout=60)
        response.raise_for_status()
        text = response.json()["candidates"][0]["content"]["parts"][0]["text"]
        parsed = json.loads(re.sub(r"^```json\s*|\s*```$", "", text.strip()))
        height, width = np.asarray(image).shape[:2]
        result=[]
        for item in parsed.get("detections", []):
            box=item.get("box", [])
            if len(box)!=4: continue
            x1,y1,x2,y2=[max(0.0,float(v)) for v in box]
            if x2<=x1 or y2<=y1: continue
            result.append(Detection(frame_second=frame_second, model_class="road_obstruction" if item.get("class")=="fallen_tree" else str(item.get("class","other")), confidence=max(0.0,min(1.0,float(item.get("confidence",0)))), bbox_xyxy=(min(x1,width),min(y1,height),min(x2,width),min(y2,height)), source_model="Gemini vision verification"))
        return result

    def scan_video(self, video_path, *, sample_every_seconds=1.0, max_frames=30):
        import cv2
        cap=cv2.VideoCapture(str(video_path)); out=[]; next_sample=0; count=0
        try:
            while count<max_frames:
                ok, frame=cap.read()
                if not ok: break
                ts=cap.get(cv2.CAP_PROP_POS_MSEC)/1000
                if ts<next_sample: continue
                out.extend(self.predict_image(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB), frame_second=ts)); count+=1; next_sample=ts+sample_every_seconds
        finally: cap.release()
        return out
