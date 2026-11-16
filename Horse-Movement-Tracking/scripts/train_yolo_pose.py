"""Fine-tune a YOLO-pose model on your own horse keypoint annotations.

Useful when you need joints the AP-10K layout does not have (carpus,
fetlock, hock, poll, withers…) or better accuracy on your camera setup.

1. Annotate frames (e.g. CVAT / Label Studio, export "YOLO pose") using the
   keypoint order of your layout (configs/horse_detailed_layout.json or AP-10K).
2. Edit configs/horse_pose_dataset.yaml (paths, kpt_shape, flip_idx).
3. python scripts/train_yolo_pose.py --data configs/horse_pose_dataset.yaml
4. python -m horsetrack clip.mp4 --pose-backend yolo --pose-model runs/pose/train/weights/best.pt \
       --layout configs/horse_detailed_layout.json
"""
from __future__ import annotations

import argparse


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="configs/horse_pose_dataset.yaml")
    ap.add_argument("--model", default="yolo11s-pose.pt", help="pretrained starting point")
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default=None)
    a = ap.parse_args()

    from ultralytics import YOLO

    model = YOLO(a.model)
    model.train(data=a.data, epochs=a.epochs, imgsz=a.imgsz, batch=a.batch, device=a.device,
                fliplr=0.5, mosaic=0.5, degrees=5, patience=30)


if __name__ == "__main__":
    main()
