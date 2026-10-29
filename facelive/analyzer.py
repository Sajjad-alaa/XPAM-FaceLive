"""Face detection (YuNet) and recognition (SFace) with OpenCV, plus a head pose from the landmarks.

Frames are processed in memory only. Nothing here writes an image or a feature to disk or to a log.
"""
import math
import os
import threading
from dataclasses import dataclass, field

import cv2
import numpy as np

from . import codes
from .models import MODELS, SFACE_FILE, YUNET_FILE, model_path

# From the face's landmarks: a nose that sits left or right of the eye line means the head is turned,
# a nose that moves between the eyes and the mouth means it is tilted up or down.
NOSE_DEPTH_RATIO = 0.55   # nose tip depth / eye distance, for the yaw estimate
NEUTRAL_NOSE_POSITION = 0.55  # nose position between eyes (0) and mouth (1) when looking straight


@dataclass
class Pose:
    yaw_u: float      # nose offset from the eye centre, in eye distances (0 = straight)
    pitch_v: float    # nose position between eyes (0) and mouth (1)
    roll_deg: float   # tilt of the eye line

    @property
    def yaw_deg(self):
        return math.degrees(math.asin(max(-1.0, min(1.0, self.yaw_u / NOSE_DEPTH_RATIO))))

    @property
    def pitch_deg(self):
        return (self.pitch_v - NEUTRAL_NOSE_POSITION) * 100.0


@dataclass
class Detection:
    bbox: tuple                      # x1, y1, x2, y2 in pixels of the processed frame
    score: float
    landmarks: list                  # 5 points: eyes, nose, mouth corners
    pose: Pose
    raw: object = field(default=None, repr=False)  # the detector's row, needed to align the face

    @property
    def width(self):
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self):
        return self.bbox[3] - self.bbox[1]


def estimate_pose(landmarks):
    """Pose proxies from 5 landmarks (any left/right order)."""
    eyes = sorted(landmarks[0:2], key=lambda p: p[0])
    mouth = sorted(landmarks[3:5], key=lambda p: p[0])
    nose = landmarks[2]
    (ex1, ey1), (ex2, ey2) = eyes
    roll = math.atan2(ey2 - ey1, ex2 - ex1)
    cx, cy = (ex1 + ex2) / 2.0, (ey1 + ey2) / 2.0
    cos_r, sin_r = math.cos(-roll), math.sin(-roll)

    def straighten(point):
        dx, dy = point[0] - cx, point[1] - cy
        return dx * cos_r - dy * sin_r, dx * sin_r + dy * cos_r

    eye_dist = math.hypot(ex2 - ex1, ey2 - ey1) or 1e-6
    nose_x, nose_y = straighten(nose)
    mouth_y = (straighten(mouth[0])[1] + straighten(mouth[1])[1]) / 2.0
    span = mouth_y if abs(mouth_y) > 1e-6 else 1e-6
    return Pose(yaw_u=nose_x / eye_dist, pitch_v=nose_y / span, roll_deg=math.degrees(roll))


def detection_from_row(row):
    row = np.asarray(row, dtype=np.float32).reshape(-1)
    x, y, w, h = [float(v) for v in row[0:4]]
    points = [(float(row[4 + 2 * i]), float(row[5 + 2 * i])) for i in range(5)]
    return Detection(
        bbox=(x, y, x + w, y + h),
        score=float(row[14]),
        landmarks=points,
        pose=estimate_pose(points),
        raw=row.copy(),
    )


class FaceAnalyzer:
    """YuNet + SFace. OpenCV objects keep per-call state, so each thread gets its own pair."""

    def __init__(self, model_dir, detect_score=0.85):
        cv2.setNumThreads(max(1, min(8, int(os.environ.get("FACELIVE_CV_THREADS", "2") or 2))))
        self._yunet = model_path(model_dir, YUNET_FILE)
        self._sface = model_path(model_dir, SFACE_FILE)
        self._score = float(detect_score)
        self._local = threading.local()

    def _tools(self):
        local = self._local
        if not hasattr(local, "detector"):
            local.detector = cv2.FaceDetectorYN.create(self._yunet, "", (320, 320), self._score, 0.3, 50)
            local.recognizer = cv2.FaceRecognizerSF.create(self._sface, "")
        return local.detector, local.recognizer

    def detect(self, frame):
        detector, _ = self._tools()
        height, width = frame.shape[:2]
        detector.setInputSize((width, height))
        _, rows = detector.detect(frame)
        if rows is None:
            return []
        found = [detection_from_row(row) for row in rows]
        found.sort(key=lambda d: d.width * d.height, reverse=True)
        return found

    def embed(self, frame, detection):
        """The SFace feature of one detected face (128 numbers, length 1)."""
        _, recognizer = self._tools()
        aligned = recognizer.alignCrop(frame, detection.raw)
        return codes.unit(recognizer.feature(aligned))

    def info(self):
        return {
            "opencv": cv2.__version__,
            "detector": YUNET_FILE,
            "recognizer": SFACE_FILE,
            "checksums": {name: item["sha256"] for name, item in MODELS.items()},
        }
