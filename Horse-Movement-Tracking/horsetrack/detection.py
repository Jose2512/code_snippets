"""Horse / person detection, horse tracking and rider association.

Uses an Ultralytics YOLO model trained on COCO (class 17 = horse,
class 0 = person) with ByteTrack so each horse keeps an ID across frames.
A segmentation model (``*-seg.pt``) is recommended because the rider's
silhouette is used to reject horse keypoints that are actually hidden
behind the rider's legs.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

COCO_PERSON = 0
COCO_HORSE = 17


@dataclass
class Detection:
    box: np.ndarray  # xyxy, float32, full-frame pixels
    conf: float
    cls: int
    track_id: int | None = None
    polygon: np.ndarray | None = None  # (N, 2) segmentation outline, if available
    _mask: np.ndarray | None = field(default=None, repr=False)

    @property
    def area(self) -> float:
        x1, y1, x2, y2 = self.box
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)

    def mask(self, shape: tuple[int, int]) -> np.ndarray:
        """Binary mask (H, W). Falls back to the box when there is no polygon."""
        if self._mask is None or self._mask.shape != shape:
            m = np.zeros(shape, dtype=np.uint8)
            if self.polygon is not None and len(self.polygon) >= 3:
                cv2.fillPoly(m, [self.polygon.astype(np.int32)], 1)
            else:
                x1, y1, x2, y2 = self.box.astype(int)
                m[max(y1, 0):max(y2, 0), max(x1, 0):max(x2, 0)] = 1
            self._mask = m.astype(bool)
        return self._mask


@dataclass
class FrameDetections:
    horses: list[Detection]
    persons: list[Detection]


def box_iou(a: np.ndarray, b: np.ndarray) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return float(inter / ua) if ua > 0 else 0.0


def _intersection_over_first(a: np.ndarray, b: np.ndarray) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area = (a[2] - a[0]) * (a[3] - a[1])
    return float(inter / area) if area > 0 else 0.0


class HorseDetector:
    """Runs YOLO detection + ByteTrack on each frame."""

    def __init__(
        self,
        weights: str = "yolo11n-seg.pt",
        device: str | None = None,
        conf: float = 0.25,
        imgsz: int = 640,
        tracker: str = "bytetrack.yaml",
    ) -> None:
        from ultralytics import YOLO  # imported lazily: heavy dependency

        self.model = YOLO(weights)
        self.device = device
        self.conf = conf
        self.imgsz = imgsz
        self.tracker = tracker

    def __call__(self, frame: np.ndarray) -> FrameDetections:
        res = self.model.track(
            frame,
            persist=True,
            classes=[COCO_PERSON, COCO_HORSE],
            conf=self.conf,
            imgsz=self.imgsz,
            tracker=self.tracker,
            device=self.device,
            verbose=False,
        )[0]
        horses: list[Detection] = []
        persons: list[Detection] = []
        if res.boxes is None or len(res.boxes) == 0:
            return FrameDetections(horses, persons)
        xyxy = res.boxes.xyxy.cpu().numpy().astype(np.float32)
        confs = res.boxes.conf.cpu().numpy()
        clss = res.boxes.cls.cpu().numpy().astype(int)
        ids = res.boxes.id.cpu().numpy().astype(int) if res.boxes.id is not None else [None] * len(xyxy)
        polys = res.masks.xy if getattr(res, "masks", None) is not None else [None] * len(xyxy)
        for box, c, k, tid, poly in zip(xyxy, confs, clss, ids, polys):
            det = Detection(box=box, conf=float(c), cls=int(k), track_id=None if tid is None else int(tid),
                            polygon=None if poly is None else np.asarray(poly, dtype=np.float32))
            (horses if k == COCO_HORSE else persons).append(det)
        return FrameDetections(horses, persons)


class TargetSelector:
    """Keeps following one horse across frames.

    * If ``horse_id`` is given, only that track is followed.
    * Otherwise the most prominent horse (area x confidence) is picked and
      followed by track ID; if the ID changes (tracker re-init after
      occlusion) the horse with the best overlap with the last box is adopted.
    """

    def __init__(self, horse_id: int | None = None, max_lost: int = 30) -> None:
        self.fixed_id = horse_id
        self.current_id: int | None = horse_id
        self.last_box: np.ndarray | None = None
        self.lost = 0
        self.max_lost = max_lost

    def update(self, horses: list[Detection]) -> Detection | None:
        chosen: Detection | None = None
        if self.fixed_id is not None:
            chosen = next((h for h in horses if h.track_id == self.fixed_id), None)
        elif horses:
            chosen = next((h for h in horses if h.track_id is not None and h.track_id == self.current_id), None)
            if chosen is None and self.last_box is not None and self.lost <= self.max_lost:
                best = max(horses, key=lambda h: box_iou(h.box, self.last_box))
                if box_iou(best.box, self.last_box) > 0.3:
                    chosen = best
            if chosen is None and (self.last_box is None or self.lost > self.max_lost):
                chosen = max(horses, key=lambda h: h.area * h.conf)
        if chosen is None:
            self.lost += 1
            return None
        self.current_id = chosen.track_id if self.fixed_id is None else self.fixed_id
        self.last_box = chosen.box.copy()
        self.lost = 0
        return chosen


def find_rider(horse: Detection, persons: list[Detection]) -> Detection | None:
    """Return the person that is mounted on ``horse`` (or None).

    A mounted rider sits on the horse's back: their box overlaps the horse
    box, their centre is horizontally inside it, they occupy the upper part
    of the horse box and their feet are clearly above the ground. A handler
    walking next to the horse (feet on the ground) is *not* a rider.
    """
    hx1, hy1, hx2, hy2 = horse.box
    hh = hy2 - hy1
    best, best_score = None, 0.0
    for p in persons:
        px1, py1, px2, py2 = p.box
        cx, cy = (px1 + px2) / 2, (py1 + py2) / 2
        overlap = _intersection_over_first(p.box, horse.box)
        centre_inside = hx1 <= cx <= hx2
        upper = cy < hy1 + 0.55 * hh
        feet_off_ground = py2 < hy2 - 0.2 * hh  # stirrups are around the belly line
        if overlap >= 0.25 and centre_inside and upper and feet_off_ground:
            score = overlap * p.conf
            if score > best_score:
                best, best_score = p, score
    return best


def smooth_flags(flags: np.ndarray, window: int) -> np.ndarray:
    """Majority vote over a centred window (removes single-frame flicker)."""
    flags = np.asarray(flags, bool)
    if window <= 1 or len(flags) == 0:
        return flags
    k = np.ones(window | 1)
    votes = np.convolve(flags.astype(float), k, mode="same")
    counts = np.convolve(np.ones(len(flags)), k, mode="same")
    return votes / counts > 0.5
