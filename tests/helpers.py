"""Helpers for the tests: images, settings and a fake analyzer."""
import dataclasses
import os
import random

import cv2
import numpy as np

from facelive import codes
from facelive.analyzer import Detection, Pose
from facelive.config import load_settings

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
MODEL_DIR = os.environ.get("FACELIVE_MODEL_DIR", "/tmp/facelive-models")


def models_available():
    return all(os.path.isfile(os.path.join(MODEL_DIR, n)) for n in (
        "face_detection_yunet_2023mar.onnx", "face_recognition_sface_2021dec.onnx"))


def settings(**changes):
    base = load_settings()
    return dataclasses.replace(base, model_dir=MODEL_DIR, **changes)


def photo(name, width=480):
    image = cv2.imread(os.path.join(DATA, name + ".jpg"))
    height = int(round(image.shape[0] * width / image.shape[1]))
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def live_variant(image, rng):
    """The same scene as a real camera sends it: new sensor noise and a pixel or two of shift each time."""
    dx, dy = rng.randint(-2, 2), rng.randint(-2, 2)
    moved = np.roll(np.roll(image, dx, axis=1), dy, axis=0).astype(np.int16)
    noise = np.random.default_rng(rng.randint(0, 2 ** 31)).normal(0, 3, image.shape)
    return np.clip(moved + noise, 0, 255).astype(np.uint8)


def jpeg(image, quality=80):
    ok, buffer = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    assert ok
    return buffer.tobytes()


class FakeAnalyzer:
    """Returns scripted faces so the challenge logic can be tested without a camera."""

    def __init__(self, feature=None):
        self.poses = []          # list of (yaw_u, pitch_v) or None per frame, consumed in order
        self.feature = codes.unit(feature if feature is not None else np.ones(128))
        self.features = []       # optional per-embed override queue
        self.faces = 1
        self.detect_calls = 0

    def detect(self, frame):
        self.detect_calls += 1
        item = self.poses.pop(0) if self.poses else (0.0, 0.55)
        if item is None:
            return []
        yaw_u, pitch_v = item
        height, width = frame.shape[:2]
        w = h = 0.3 * width
        result = []
        for index in range(self.faces):
            x1, y1 = width * (0.1 + 0.45 * index), (height - h) / 2
            landmarks = [(x1 + w * 0.3, y1 + h * 0.35), (x1 + w * 0.7, y1 + h * 0.35),
                         (x1 + w * 0.5, y1 + h * 0.5), (x1 + w * 0.35, y1 + h * 0.75),
                         (x1 + w * 0.65, y1 + h * 0.75)]
            pose = Pose(yaw_u=yaw_u, pitch_v=pitch_v, roll_deg=0.0)
            result.append(Detection(bbox=(x1, y1, x1 + w, y1 + h), score=0.97,
                                    landmarks=landmarks, pose=pose, raw=None))
        return result

    def embed(self, frame, detection):
        if self.features:
            return codes.unit(self.features.pop(0))
        return self.feature


def noisy_frame(rng, size=(480, 360)):
    """A different picture each time (so the replay guard sees a live camera)."""
    base = np.full((size[1], size[0], 3), 128, dtype=np.uint8)
    noise = np.random.default_rng(rng.randint(0, 2 ** 31)).normal(0, 12, base.shape)
    return np.clip(base + noise, 0, 255).astype(np.uint8)


class Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now

    def tick(self, seconds=0.25):
        self.now += seconds
        return self.now


def rng(seed=1):
    return random.Random(seed)


def face_crop(name, analyzer, width=480):
    """The photo cropped around the face (as if the person sat closer), for photos taken from far away."""
    image = cv2.imread(os.path.join(DATA, name + ".jpg"))
    face = analyzer.detect(image)[0]
    x1, y1, x2, y2 = face.bbox
    side = max(x2 - x1, y2 - y1) * 2.4
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    left, top = int(max(0, cx - side / 2)), int(max(0, cy - side / 2))
    crop = image[top:int(top + side), left:int(left + side)]
    return cv2.resize(crop, (width, int(round(crop.shape[0] * width / crop.shape[1]))), interpolation=cv2.INTER_AREA)
