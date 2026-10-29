"""Download the two OpenCV zoo models into a folder and check their pinned SHA-256 (used at image build time).

    python tools/fetch_models.py /opt/facelive/models
"""
import os
import sys
import urllib.request

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from facelive.models import MODELS, sha256_of  # noqa: E402


def main(folder):
    os.makedirs(folder, exist_ok=True)
    for name, item in MODELS.items():
        target = os.path.join(folder, name)
        urllib.request.urlretrieve(item["url"], target)
        found = sha256_of(target)
        if found != item["sha256"]:
            os.unlink(target)
            raise SystemExit("checksum mismatch for %s: expected %s, got %s" % (name, item["sha256"], found))
        print("ok", name, found)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    main(sys.argv[1])
