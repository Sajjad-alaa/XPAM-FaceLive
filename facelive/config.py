"""Settings of the engine, read from the environment.

The FACE_* names are the same ones core uses (config.txt), so one setting drives both.
"""
import os
from dataclasses import dataclass


def _text(name, default=""):
    return os.environ.get(name, default).strip()


def _number(name, default, low, high, cast=float):
    raw = _text(name)
    try:
        value = cast(raw) if raw else default
    except ValueError:
        value = default
    return max(low, min(high, value))


def _flag(name, default=False):
    raw = _text(name).lower()
    if not raw:
        return default
    return raw in ("1", "true", "yes", "on")


@dataclass(frozen=True)
class Settings:
    host: str
    port: int
    core_host: str
    core_tls_verify: bool
    bootstrap_token: str
    name: str
    access_key_file: str
    model_dir: str
    http_timeout: float
    # capture rules
    enrollment_samples: int
    frame_max_bytes: int
    frame_min_interval: float
    frame_interval_ms: int
    capture_timeout: float
    min_face_ratio: float
    min_face_size: int
    detect_score: float
    debug: bool
    liveness_mode: str
    challenge_timeout: float
    yaw_turn: float
    pitch_nod: float
    # limits
    max_connections: int
    max_connections_per_ip: int
    connects_per_minute: int
    workers: int


def load_settings():
    mode = _text("FACE_LIVENESS_MODE", "motion").lower()
    return Settings(
        host=_text("FACELIVE_HOST", "0.0.0.0"),
        port=_number("FACELIVE_PORT", 8085, 1, 65535, int),
        core_host=_text("CORE_HOST", "http://core:8080").rstrip("/"),
        core_tls_verify=_flag("FACELIVE_CORE_TLS_VERIFY", True),
        bootstrap_token=_text("BOOTSTRAP_TOKEN"),
        name=_text("FACELIVE_NAME", "") or default_name(),
        access_key_file=_text("FACELIVE_ACCESS_KEY_FILE", "/opt/facelive/data/keys/.access_key"),
        model_dir=_text("FACELIVE_MODEL_DIR", "/opt/facelive/models"),
        http_timeout=_number("FACELIVE_HTTP_TIMEOUT", 8.0, 1.0, 60.0),
        enrollment_samples=_number("FACE_RECOGNITION_ENROLLMENT_SAMPLES", 3, 1, 10, int),
        frame_max_bytes=_number("FACE_RECOGNITION_FRAME_MAX_BYTES", 2 * 1024 * 1024, 16 * 1024, 8 * 1024 * 1024, int),
        frame_min_interval=_number("FACE_RECOGNITION_FRAME_MIN_INTERVAL", 0.18, 0.02, 5.0),
        frame_interval_ms=_number("FACELIVE_FRAME_INTERVAL_MS", 250, 120, 2000, int),
        capture_timeout=_number("FACE_RECOGNITION_CAPTURE_TIMEOUT", 30, 5, 120.0),
        min_face_ratio=_number("FACE_RECOGNITION_MIN_FACE_RATIO", 0.06, 0.005, 0.5),
        min_face_size=_number("FACE_LIVENESS_MIN_FACE_SIZE", 96, 32, 400, int),
        detect_score=_number("FACELIVE_DETECT_SCORE", 0.85, 0.5, 0.99),
        debug=_flag("FACE_RECOGNITION_DEBUG", False),
        # "off" = no head-movement challenge; anything else (motion, onnx, hybrid) = challenge
        liveness_mode="off" if mode == "off" else "challenge",
        challenge_timeout=_number("FACE_LIVENESS_CHALLENGE_TIMEOUT", 15, 5, 60.0),
        # how far the head must move (see liveness.py); tune only after watching real cameras
        yaw_turn=_number("FACELIVE_YAW_TURN", 0.15, 0.05, 0.4),
        pitch_nod=_number("FACELIVE_PITCH_NOD", 0.09, 0.03, 0.3),
        max_connections=_number("FACELIVE_MAX_CONNECTIONS", 40, 1, 500, int),
        max_connections_per_ip=_number("FACELIVE_MAX_CONNECTIONS_PER_IP", 4, 1, 100, int),
        connects_per_minute=_number("FACELIVE_CONNECTS_PER_MINUTE", 30, 1, 1000, int),
        workers=_number("FACELIVE_WORKERS", 2, 1, 16, int),
    )


def default_name():
    import socket

    host = socket.gethostname() or "localhost"
    return ("[Facelive]-" + host)[:128]
