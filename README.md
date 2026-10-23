# RoadGuard AI

RoadGuard AI is a road-hazard reporting and review app. It separates model suggestions from admin-confirmed public incidents. The local Docker stack runs a CPU-only YOLOv8s pothole model over uploaded images and sampled prerecorded video frames. The API still uses in-memory incident records and local video storage, so it is not a production data backend.

## What exists

- React + Vite commuter/admin interface with browser GPS, OpenStreetMap centering, nearby published-hazard filtering, weather and review controls.
- FastAPI API with explicit incident lifecycle, public filtering, reports, health, analytics, and integration status.
- SQL migration with core tables and RLS policies; optional demo seed data.
- Docker configuration for Redis-backed video jobs and a persistent inference worker.
- OpenStreetMap geocoding and OSRM route planning, including a typed start-place fallback when browser GPS is unavailable.

## Run locally

Backend: `cd backend && python -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt && uvicorn app.main:app --reload`

Frontend: `cd frontend && npm install && npm run dev`

Or run the local containers with `docker compose up --build` (frontend at `http://localhost:8080`, API at `http://localhost:8001`). The first worker startup downloads CPU-only inference dependencies and both checkpoints. On Apple Silicon, the worker uses Linux amd64 emulation and can be slower. The dashboard accepts JPG, PNG and WEBP images or prerecorded MP4, MOV and WEBM clips. Geotagged still images with detections publish approximate camera-location alerts to route users on the same running API; images without GPS and all video results remain detections only. The API stores incidents in memory and media on local disk.

## Vercel deployment

The frontend and a separate FastAPI function are deployed as the `roadguard-ai` and `roadguard-api` Vercel projects. The Vercel frontend retains SSO protection. Both versioned Supabase migrations have been applied to the `roadguard` project; the public schema contains the RoadGuard tables with RLS enabled. The API currently reports `demo-memory` and `local-disk`; it does not yet use Supabase as its incident repository.

Vercel Functions cannot provide the local Redis queue, persistent worker or durable video files used for model inference. Video processing is verified in the local Docker stack. The Vercel API has no worker heartbeat, so hosted video inference remains unavailable until the worker and shared persistent storage run on a persistent compute host. The latest server-side route proxy and manual-start planner changes are local until the Vercel team scope is reauthorized and those projects are redeployed.

The UI requests live device location to center OpenStreetMap, find nearby reports, and request current weather from Open-Meteo. If GPS permission is blocked, enter a starting place and destination in the route planner. Route planning uses the API proxy with OpenStreetMap Nominatim and OSRM. Map attribution is shown in the UI. Nominatim is intended for low-volume lookups. Set `VITE_API_URL` to point the frontend to the API. Admin review and upload actions are available without application sign-in.

## Integration and deployment checklist

| Provider | Purpose / setup | Secret location | Optional / cost |
| --- | --- | --- | --- |
| Supabase | PostgreSQL schema is deployed; current API incidents remain in demo memory | Backend `SUPABASE_URL`, `SUPABASE_PUBLISHABLE_KEY`; frontend public URL/key | Optional for local demo; hosted usage may cost |
| Map | OpenStreetMap centered on browser GPS; map is for location context, not navigation | None | Public map tile service terms and attribution apply |
| Weather provider | Open-Meteo current conditions by browser GPS | None for non-commercial use | Free tier is non-commercial only; commercial use requires another plan/provider |
| Road-damage models | YOLOv8s pothole detector plus the user's YOLOv8n pothole/crack/waterlogging model; see `docs/model-inference.md` for benchmark and license limits | Worker environment only | Worker compute and persistent hosting may cost; auxiliary dataset-derived weights are non-commercial |
| Hosting | Frontend static host and persistent API/worker host | Provider secret manager / CLI login | Hosting may cost |

Do not put service-role keys or model files in the frontend or Git. The `.env` file is ignored. The API now has a Supabase insert/read path, enabled by `SUPABASE_SERVICE_ROLE_KEY` in the server environment; without it, automatic alerts stay in local memory. Persistent media storage and worker compute are still needed for production. Camera GPS is an approximate photo viewpoint, not a surveyed coordinate for each detected object.

## API

See `docs/API.md`. The in-memory repository is for local demonstration only; use the migration-backed repository before multi-instance or production deployment. Public hazards are only records in `PUBLISHED` state. Routes are filtered by distance to the supplied route polyline, not by city membership.

## Model and source notes

The Docker worker uses an Apache-2.0 YOLOv8s pothole checkpoint on uploaded still images and sampled frames from prerecorded videos. See `docs/model-inference.md` for setup, limitations, attribution and license notes. Uploaded video is historical prerecorded media, never live CCTV.

Video uploads are size-limited and container-signature checked; image uploads are limited to supported formats and checked against their file signatures. Both use generated storage names. Docker Compose includes Redis-backed media jobs and a YOLOv8 worker. The first worker start downloads the model checkpoint. This configuration is local development only; the current Vercel projects do not run this persistent worker, shared storage or durable incident repository. Those production services still need persistent hosting.
