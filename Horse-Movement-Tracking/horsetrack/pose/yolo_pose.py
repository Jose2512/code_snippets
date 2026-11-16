"""Ultralytics YOLO-pose model trained on horse keypoints.

Use this backend with weights you trained with ``scripts/train_yolo_pose.py``
(any layout, e.g. AP-10K or the detailed 27-joint equine layout).
"""
from __future__ import annotations

import numpy as np

from ..detection import box_iou
from ..layouts import KeypointLayout
from .base import PoseBackend


class YoloPose(PoseBackend):
    def __init__(self, model_path: str, layout: KeypointLayout, device: str | None = None,
                 conf: float = 0.2, imgsz: int = 640) -> None:
        from ultralytics import YOLO

        self.model = YOLO(model_path)
        self.layout = layout
        self.device = device
        self.conf = conf
        self.imgsz = imgsz

    def estimate(self, frame: np.ndarray, box: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        r = self.model(frame, conf=self.conf, imgsz=self.imgsz, device=self.device, verbose=False)[0]
        if r.keypoints is None or len(r.boxes) == 0:
            return self._empty()
        boxes = r.boxes.xyxy.cpu().numpy()
        ious = [box_iou(b, box) for b in boxes]
        i = int(np.argmax(ious))
        if ious[i] < 0.1:
            return self._empty()
        xy = r.keypoints.xy[i].cpu().numpy().astype(np.float32)
        if xy.shape[0] != self.layout.num_keypoints:
            raise ValueError(f"YOLO pose model has {xy.shape[0]} keypoints, layout expects {self.layout.num_keypoints}")
        conf = (r.keypoints.conf[i].cpu().numpy() if r.keypoints.conf is not None
                else np.ones(len(xy))).astype(np.float32)
        return xy, conf
