# API contract (starter)

Base URL: `http://localhost:8000`. OpenAPI is available at `/docs`.

| Method | Path | Behavior |
| --- | --- | --- |
| GET | `/api/health` | Component readiness, explicitly reporting unconfigured providers |
| GET | `/api/incidents` | Admin-only lifecycle queue; optional `status` filter |
| GET | `/api/incidents/{id}` | Publicly readable only when published; unpublished records require admin access |
| POST | `/api/videos` | Admin-only, size-limited prerecorded upload with filename sanitization and basic container-signature check |
| POST | `/api/images` | Admin-only JPEG, PNG or WEBP upload with size and file-signature checks |
| GET | `/api/videos` / `/api/videos/{id}` | Admin-only uploaded-video metadata |
| POST | `/api/videos/{id}/process` | Admin-only; queues an uploaded image or prerecorded video for the Redis-backed YOLOv8 pothole worker |
| GET | `/api/jobs/{id}` | Admin-only inference job status and unverified frame detections |
| POST | `/api/jobs/{id}/cancel` | Admin-only cancellation of a queued job |
| GET | `/api/hazards/nearby?latitude=&longitude=&radius_m=` | Published hazards only |
| POST | `/api/hazards/route-query` | Published hazards within `corridor_m` of supplied `[lat,lon]` route vertices |
| POST | `/api/route/plan` | Geocodes a typed start/destination or resolves a destination from live GPS, then returns OSRM route geometry and turn-by-turn URL |
| POST | `/api/reports` | Create pending user report, no automatic publication |
| GET | `/api/reports` | Admin-only report list |
| PATCH | `/api/incidents/{id}/review` | Admin lifecycle transition via `decision` and `notes` |
| GET | `/api/analytics/summary` | Labeled demo summary |
| GET | `/api/integrations/status` | Configured/not configured integration flags |

Review and media administration endpoints are public in the deployed app and do not require an `X-Admin-Token` or Supabase session. This is the requested no-login configuration. Supabase client settings still support public commuter hazard reads; the API has not yet been migrated to persistent Supabase incident storage.

Video inference must run with the Docker API, Redis queue, and `video-worker` service described in `docker-compose.yml`. The Vercel API deployment uses ephemeral files and has no Redis worker, so video upload and inference are not available there. The API's incident/report dictionaries are demo memory and reset on process restart; configure a durable repository before relying on operational history.
