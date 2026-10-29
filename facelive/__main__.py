import asyncio
import logging
import signal
import sys

from .analyzer import FaceAnalyzer
from .config import load_settings
from .server import FaceServer


def check():
    """Build-time smoke test: the models load, check out and see a face-less picture as empty."""
    import numpy as np

    settings = load_settings()
    analyzer = FaceAnalyzer(settings.model_dir, settings.detect_score)
    faces = analyzer.detect(np.full((360, 480, 3), 128, dtype=np.uint8))
    assert faces == [], "unexpected face in a blank picture"
    print("facelive check ok", analyzer.info()["opencv"])


async def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("websockets").setLevel(logging.WARNING)  # (its connection logs carry the request path)
    settings = load_settings()
    analyzer = FaceAnalyzer(settings.model_dir, settings.detect_score)  # (checks the model checksums)
    logging.getLogger("facelive").info("models loaded: %s", analyzer.info())
    server = FaceServer(settings, analyzer)
    loop = asyncio.get_running_loop()
    stop = asyncio.Event()
    for name in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(name, stop.set)
    background = asyncio.create_task(server.core.keep_connected())
    running = asyncio.create_task(server.run())
    await stop.wait()
    running.cancel()
    background.cancel()


if __name__ == "__main__":
    if sys.argv[1:] == ["--check"]:
        check()
    else:
        asyncio.run(main())
