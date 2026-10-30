# Road hazard inference

RoadGuard uses a multi-model detector ensemble for uploaded images and sampled prerecorded video. Gemini is not part of the active inference path.

## Processing order

1. **Pothole YOLOv8 model** detects road-surface damage using the trained checkpoint.
2. **Multi-Hazard YOLOv8 model** detects asphalt cracks and surface waterlogging.
3. **Infrastructure YOLO-World zero-shot model** detects road assets, traffic cones, and infrastructure damage.
4. **Grounding DINO Obstruction model** (`IDEA-Research/grounding-dino-tiny`) detects fallen trees, tree trunks, branches across lanes, downed power lines, and physical roadway blocks.
5. **Specialist Tree Obstruction detector** targets fallen timber, logs, and blocking boughs with tuned Grounding DINO cross-attention.
6. **Ensemble merge and NMS** combines same-class and overlapping proposal boxes and returns prioritized model suggestions for review. All AI detections remain unverified until a human reviews them.

## Runtime

- No Gemini API key is required.
- The worker loads each model independently. A failed checkpoint is reported in the job's `model_errors`; the remaining available models continue to run.
- Model weights are cached in the Docker `model-cache` volume. The worker also mounts `./data/model-cache/weights` at `/app/weights` for local checkpoints.
- Still images are analyzed once. Videos are sampled every two seconds by default, up to 12 frames, and the same frames are reused across available models.
- `ROAD_DAMAGE_CONFIDENCE` defaults to `0.15`; `ROAD_OBSTRUCTION_CONFIDENCE` defaults to `0.15`; `ROAD_INFRASTRUCTURE_CONFIDENCE` defaults to `0.04`.

## Map behavior and limitations

A still image with EXIF GPS can create pending review suggestions at the approximate camera location. These appear as amber pins on the live map when they are in the map's local area. The marker is not the surveyed position of each defect. Images without GPS and video detections are not assigned a map position. AI suggestions remain pending review and do not become public route warnings until verified and published by an administrator.

The local worker and media storage require the Docker API, Redis queue, and `video-worker` service. Vercel Functions do not provide this persistent worker or shared local media volume; hosted media inference requires persistent compute and storage.
