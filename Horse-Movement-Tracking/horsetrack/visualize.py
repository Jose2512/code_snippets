"""Drawing of the skeleton, joint angles, head vector, hoof trails and HUD."""
from __future__ import annotations

from collections import deque

import cv2
import numpy as np

from .layouts import KeypointLayout

# BGR colours per body part
PART_COLORS = {
    "LF": (60, 200, 255),   # amber
    "RF": (255, 160, 40),   # blue
    "LH": (80, 220, 80),    # green
    "RH": (220, 80, 220),   # magenta
    "axial": (240, 240, 240),
}
HORSE_BOX = (90, 200, 90)
RIDER_BOX = (40, 140, 255)
HEAD_COLOR = (0, 230, 255)


def _pt(p) -> tuple[int, int]:
    return int(round(float(p[0]))), int(round(float(p[1])))


def _ok(p) -> bool:
    return bool(np.all(np.isfinite(p)))


def _label(img, text, org, color=(255, 255, 255), scale=0.5, thick=1):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 2, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


class Renderer:
    def __init__(self, layout: KeypointLayout, fps: float, trail_s: float = 0.8, frame_height: int = 720):
        self.layout = layout
        self.trails = {k: deque(maxlen=max(2, int(trail_s * fps))) for k in layout.hooves}
        self.s = max(0.6, frame_height / 720)  # scale line widths / text to resolution

    def draw(self, frame: np.ndarray, xy: np.ndarray, detected: np.ndarray, metrics: dict,
             horse_box=None, rider_box=None, hud: dict | None = None) -> np.ndarray:
        img = frame
        L, s = self.layout, self.s
        lw, r = max(1, int(2 * s)), max(2, int(4 * s))

        if horse_box is not None and _ok(horse_box):
            x1, y1, x2, y2 = map(int, horse_box)
            cv2.rectangle(img, (x1, y1), (x2, y2), HORSE_BOX, 1)
            _label(img, "horse", (x1, max(12, y1 - 4)), HORSE_BOX, 0.45 * s)
        if rider_box is not None and _ok(rider_box):
            x1, y1, x2, y2 = map(int, rider_box)
            cv2.rectangle(img, (x1, y1), (x2, y2), RIDER_BOX, 1)
            _label(img, "rider", (x1, max(12, y1 - 4)), RIDER_BOX, 0.45 * s)

        # hoof trails
        for limb, hoof in L.hooves.items():
            p = xy[L.idx(hoof)]
            self.trails[limb].append(_pt(p) if _ok(p) else None)
            pts = list(self.trails[limb])
            for a, b in zip(pts, pts[1:]):
                if a and b:
                    cv2.line(img, a, b, PART_COLORS[limb], max(1, lw - 1), cv2.LINE_AA)

        # skeleton
        for a, b in L.skeleton:
            pa, pb = xy[L.idx(a)], xy[L.idx(b)]
            if _ok(pa) and _ok(pb):
                part = L.part_of(b) if L.part_of(b) != "axial" else L.part_of(a)
                cv2.line(img, _pt(pa), _pt(pb), PART_COLORS[part], lw, cv2.LINE_AA)

        # joints (filled = detected, hollow = interpolated)
        for i, name in enumerate(L.keypoints):
            p = xy[i]
            if not _ok(p):
                continue
            col = PART_COLORS[L.part_of(name)]
            cv2.circle(img, _pt(p), r, col, -1 if detected[i] else lw, cv2.LINE_AA)
            if detected[i]:
                cv2.circle(img, _pt(p), r, (0, 0, 0), 1, cv2.LINE_AA)

        # joint angle values
        for ang in L.angles:
            v = metrics.get(f"angle_{ang.name}")
            p = xy[L.idx(ang.vertex)]
            if v is not None and np.isfinite(v) and _ok(p):
                _label(img, f"{v:.0f}", (_pt(p)[0] + r + 2, _pt(p)[1] - r), PART_COLORS[L.part_of(ang.vertex)], 0.42 * s)

        # head vector
        nose, neck = xy[L.idx(L.nose)], xy[L.idx(L.neck)]
        if _ok(nose) and _ok(neck):
            cv2.arrowedLine(img, _pt(neck), _pt(nose), HEAD_COLOR, lw + 1, cv2.LINE_AA, tipLength=0.15)

        if hud:
            self._hud(img, hud)
        return img

    def _hud(self, img, hud: dict) -> None:
        s = self.s
        lines = [f"{k}: {v}" for k, v in hud.items()]
        lh = int(20 * s)
        w = int(max(len(t) for t in lines) * 8.5 * s) + 16
        overlay = img.copy()
        cv2.rectangle(overlay, (8, 8), (8 + w, 14 + lh * len(lines)), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.55, img, 0.45, 0, dst=img)
        for i, t in enumerate(lines):
            _label(img, t, (16, 8 + lh * (i + 1)), (255, 255, 255), 0.5 * s)
