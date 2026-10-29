"""Face codes: "xpamf1:" + base64 of 128 little-endian float32 numbers, scaled to length 1.

Core reads the same format (apps/users/face_sface.py). A code carries no image.
"""
import base64

import numpy as np

CODE_PREFIX = "xpamf1:"
FEATURE_SIZE = 128


def unit(vector):
    vector = np.asarray(vector, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(vector))
    if vector.size != FEATURE_SIZE or not np.isfinite(norm) or norm <= 0:
        raise ValueError("not a face feature")
    return (vector / norm).astype(np.float32)


def mean_feature(vectors):
    """The average of several features, scaled to length 1."""
    return unit(np.mean([unit(v) for v in vectors], axis=0))


def cosine(a, b):
    return float(np.dot(unit(a), unit(b)))


def encode_code(vector):
    return CODE_PREFIX + base64.b64encode(unit(vector).astype("<f4").tobytes()).decode("ascii")


def decode_code(code):
    if not isinstance(code, str) or not code.startswith(CODE_PREFIX):
        raise ValueError("not a face code")
    raw = base64.b64decode(code[len(CODE_PREFIX):], validate=True)
    if len(raw) != FEATURE_SIZE * 4:
        raise ValueError("face code has the wrong size")
    return unit(np.frombuffer(raw, dtype="<f4"))
