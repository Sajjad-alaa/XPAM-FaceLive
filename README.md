# xpam-facelive

Face enrolment and face sign-in checks for XPAM. The Workspace page `/workspace/facelive/capture` opens a
WebSocket to `/ws/facelive/capture/?token=...&mode=capture`; this service answers it.

- Detection: OpenCV YuNet (MIT). Recognition: OpenCV SFace (Apache-2.0). See `NOTICE`.
- Python 3.11 in the image (the code also runs on 3.9 for tests). Dependencies: `requirements.txt`.
- The service is a core **service account**: it registers once with `BOOTSTRAP_TOKEN`
  (`POST /api/v1/terminal/terminal-registrations/`, type `facelive`), keeps the access key in
  `FACELIVE_ACCESS_KEY_FILE`, and signs every request (HTTP Signature, hmac-sha256).
- Core calls used: `GET face/engine/context/` (what the token is for), `POST face/engine/compare/` (is this
  the user), `POST face/callback/` (the result and the face code). All under `/api/v1/authentication/`.

## Flow

1. Look straight (3 frames) -> 2. a random head movement (turn left and right, or nod) -> 3. enrol: 3 clear
samples averaged into one template (`xpamf1:` code); sign in: 2 frames that core says match the user.
Frames, features and codes are processed in memory only and never logged or written.

## Settings (environment)

| Name | Default | |
|---|---|---|
| `CORE_HOST` | `http://core:8080` | core address inside the Docker network |
| `BOOTSTRAP_TOKEN` | | only for the first registration |
| `FACELIVE_ACCESS_KEY_FILE` | `/opt/facelive/data/keys/.access_key` | keep it on a volume |
| `FACELIVE_PORT` | `8085` | |
| `FACE_RECOGNITION_ENROLLMENT_SAMPLES` | `3` | same names as core's settings |
| `FACE_RECOGNITION_CAPTURE_TIMEOUT` | `30` | seconds, plus the challenge time |
| `FACE_RECOGNITION_FRAME_MAX_BYTES` / `_FRAME_MIN_INTERVAL` | 2 MB / 0.18 s | frame size and rate limit |
| `FACE_RECOGNITION_MIN_FACE_RATIO` / `FACE_LIVENESS_MIN_FACE_SIZE` | 0.06 / 96 | "move closer" rules |
| `FACE_LIVENESS_MODE` | `motion` | `off` = no head movement challenge |
| `FACE_LIVENESS_CHALLENGE_TIMEOUT` | `15` | seconds per challenge, two tries |
| `FACELIVE_YAW_TURN` / `FACELIVE_PITCH_NOD` | 0.15 / 0.09 | head movement needed (nose offset in eye distances); tune after watching real cameras |
| `FACE_RECOGNITION_DEBUG` | `false` | the page draws boxes and landmarks |
| `FACELIVE_MAX_CONNECTIONS`, `_PER_IP`, `_CONNECTS_PER_MINUTE` | 40, 4, 30 | |

Match thresholds are core's (`FACE_RECOGNITION_COSINE_THRESHOLD` 0.40, `FACE_RECOGNITION_DISTANCE_THRESHOLD`
1.0955); the engine never decides a match by itself, core does.

## Not supported

- Monitor mode (`mode=monitor`, pausing a live session when the person leaves): the page gets a fatal
  "not supported" error.
- ONNX anti-spoofing models (`FACE_LIVENESS_MODE=onnx|hybrid` run the same head-movement challenge).

## Develop and test

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements-dev.txt
python tools/fetch_models.py /tmp/facelive-models
FACELIVE_MODEL_DIR=/tmp/facelive-models PYTHONPATH=. .venv/bin/python -m pytest tests -q
```
