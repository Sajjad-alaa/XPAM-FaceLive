"""Process entry point for `python -m facelive`.

This module is compiled to a native extension (.so) in the shipped image, so the startup sequence,
the build-time self-check and the signal handling are not present as readable source. `__main__.py`
is only a thin launcher that calls run() here.
"""
import asyncio
import logging
import os
import signal
import sys

# Cap OpenCV's native image decoder before cv2 is imported so crafted headers cannot allocate huge buffers
os.environ.setdefault("OPENCV_IO_MAX_IMAGE_WIDTH", "1920")
os.environ.setdefault("OPENCV_IO_MAX_IMAGE_HEIGHT", "1920")
os.environ.setdefault("OPENCV_IO_MAX_IMAGE_PIXELS", str(1920 * 1920))

from .analyzer import FaceAnalyzer
from .config import load_settings
from .server import FaceServer


def check():
    import numpy as np

    settings = load_settings()
    analyzer = FaceAnalyzer(settings.model_dir, settings.detect_score)
    faces = analyzer.detect(np.full((360, 480, 3), 128, dtype=np.uint8))
    assert faces == [], "unexpected face in a blank picture"
    print("facelive check ok", analyzer.info()["opencv"])


async def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("websockets").setLevel(logging.WARNING)
    settings = load_settings()
    analyzer = FaceAnalyzer(settings.model_dir, settings.detect_score)
    logging.getLogger("facelive").info("models loaded: %s", analyzer.info())
    server = FaceServer(settings, analyzer)
    loop = asyncio.get_running_loop()
    stop = asyncio.Event()
    for name in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(name, stop.set)
    background = asyncio.create_task(server.core.keep_connected())
    heartbeat = asyncio.create_task(server.core.heartbeat(lambda: server.limits.active))
    running = asyncio.create_task(server.run())
    await stop.wait()
    running.cancel()
    heartbeat.cancel()
    background.cancel()


def run(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv == ["--check"]:
        check()
    else:
        asyncio.run(main())
