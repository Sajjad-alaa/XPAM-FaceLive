"""The two ONNX models and their pinned SHA-256 checksums (see NOTICE for the licences)."""
import hashlib
import os

YUNET_FILE = "face_detection_yunet_2023mar.onnx"
SFACE_FILE = "face_recognition_sface_2021dec.onnx"

# Both come from the OpenCV model zoo (https://github.com/opencv/opencv_zoo)
MODELS = {
    YUNET_FILE: {
        "sha256": "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4",
        "url": "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/" + YUNET_FILE,
    },
    SFACE_FILE: {
        "sha256": "0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79",
        "url": "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/" + SFACE_FILE,
    },
}


def sha256_of(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def model_path(model_dir, name):
    """Path of a model file after checking its checksum (a changed file is refused)."""
    path = os.path.join(model_dir, name)
    if not os.path.isfile(path):
        raise RuntimeError("model file is missing: %s" % path)
    if sha256_of(path) != MODELS[name]["sha256"]:
        raise RuntimeError("model file has an unexpected checksum: %s" % path)
    return path
