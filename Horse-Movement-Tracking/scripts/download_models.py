"""Download the default models into ./models.

* Detector: Ultralytics YOLO11 segmentation (COCO: horse + person).
* Pose: ViTPose trained on AP-10K (17 animal keypoints), ONNX export from
  the easy_ViTPose project on Hugging Face (Apache-2.0).

Usage:  python scripts/download_models.py [--size s|b|l|h]
"""
from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

HF_REPO = "JunkyByte/easy_ViTPose"
YOLO_URL = "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n-seg.pt"


def fetch(url: str, dest: Path) -> bool:
    if dest.exists():
        print(f"✓ {dest} already present")
        return True
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"↓ {url}")
    try:
        tmp = dest.with_suffix(dest.suffix + ".part")
        urllib.request.urlretrieve(url, tmp)
        tmp.replace(dest)
        print(f"✓ saved {dest} ({dest.stat().st_size / 1e6:.1f} MB)")
        return True
    except Exception as e:  # network / 404
        print(f"✗ failed: {e}")
        return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", choices="sblh", default="b", help="ViTPose size (s=fast … h=accurate)")
    ap.add_argument("--dir", default="models")
    a = ap.parse_args()
    out = Path(a.dir)

    ok = fetch(YOLO_URL, out / "yolo11n-seg.pt")
    name = f"vitpose-{a.size}-ap10k.onnx"
    url = f"https://huggingface.co/{HF_REPO}/resolve/main/onnx/ap10k/{name}"
    ok &= fetch(url, out / name)
    if not ok:
        print(f"\nSome downloads failed. Get the AP-10K ONNX model manually from "
              f"https://huggingface.co/{HF_REPO}/tree/main/onnx/ap10k and save it in {out}/.",
              file=sys.stderr)
        return 1
    print(f"\nReady:  python -m horsetrack your_clip.mp4 --detector {out}/yolo11n-seg.pt "
          f"--pose-model {out}/{name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
