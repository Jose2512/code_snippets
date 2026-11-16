"""Per-frame kinematic measures from 2-D keypoints.

Conventions (image coordinates, y grows downward):

* ``direction`` is +1 when the horse faces right in the image, -1 when it
  faces left, so every signed measure is expressed relative to the horse
  ("forward" = toward the head) whichever way it moves.
* Joint angles are interior angles in degrees (180 = fully extended).
* ``protraction_<limb>``: angle between the vertical and the line from the
  proximal joint (shoulder/hip) to the hoof. Positive = hoof ahead of the
  shoulder/hip (protraction), negative = behind (retraction).
* ``head_height``: height of the nose above the neck/withers point divided
  by body length (positive = nose above).
* ``neck_elevation``: angle of the neck->nose line above the body axis.
* ``face_inclination``: angle of the face (poll or eyes -> nose) from the
  vertical; 0 = face vertical, positive = nose in front of the vertical,
  negative = behind the vertical.

All measures are 2-D projections, so they are most meaningful when the
camera films the horse from the side (lateral view).
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .layouts import KeypointLayout

NAN = float("nan")


def joint_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    v1, v2 = a - b, c - b
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if not np.isfinite(n1 * n2) or n1 == 0 or n2 == 0:
        return NAN
    cos = float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))
    return math.degrees(math.acos(cos))


def _elev(vx: float, vy_img: float, direction: float) -> float:
    """Elevation (deg) of an image-space vector in the horse's frame:
    0 = pointing forward horizontally, +90 = straight up."""
    return math.degrees(math.atan2(-vy_img, vx * direction))


def estimate_direction(xy: np.ndarray, layout: KeypointLayout) -> np.ndarray:
    """Facing direction per frame (+1 right / -1 left), gap-filled and
    median-filtered to avoid flicker. xy is (T, K, 2)."""
    f, r = layout.idx(layout.body_axis[0]), layout.idx(layout.body_axis[1])
    dx = xy[:, f, 0] - xy[:, r, 0]
    nose = layout.idx(layout.nose)
    dx = np.where(np.isnan(dx), xy[:, nose, 0] - xy[:, r, 0], dx)
    s = pd.Series(np.sign(dx)).replace(0, np.nan).ffill().bfill().fillna(1.0)
    win = 15 if len(s) >= 15 else max(1, len(s) | 1)
    s = s.rolling(win, center=True, min_periods=1).median()
    return np.where(s >= 0, 1.0, -1.0)


def frame_metrics(p: np.ndarray, layout: KeypointLayout, direction: float) -> dict[str, float]:
    """Metrics for one frame. p is (K, 2) with NaN for missing points."""
    P = {name: p[i] for i, name in enumerate(layout.keypoints)}
    m: dict[str, float] = {}

    for ang in layout.angles:
        m[f"angle_{ang.name}"] = joint_angle(P[ang.a], P[ang.vertex], P[ang.c])

    for limb, chain in layout.limbs.items():
        top, hoof = P[chain[0]], P[chain[-1]]
        d = hoof - top
        m[f"protraction_{limb}"] = (math.degrees(math.atan2(d[0] * direction, d[1]))
                                    if np.all(np.isfinite(d)) else NAN)

    front, rear = P[layout.body_axis[0]], P[layout.body_axis[1]]
    body = front - rear
    body_len = float(np.linalg.norm(body)) if np.all(np.isfinite(body)) else NAN
    m["body_length_px"] = body_len
    m["body_pitch"] = _elev(body[0], body[1], direction) if np.isfinite(body_len) else NAN

    nose, neck = P[layout.nose], P[layout.neck]
    m["nose_x"], m["nose_y"] = float(nose[0]), float(nose[1])
    if np.all(np.isfinite(nose - neck)):
        m["head_height"] = float((neck[1] - nose[1]) / body_len) if body_len and np.isfinite(body_len) else NAN
        neck_vec = nose - neck
        m["neck_elevation"] = (_elev(neck_vec[0], neck_vec[1], direction) - m["body_pitch"]
                               if np.isfinite(m["body_pitch"]) else _elev(neck_vec[0], neck_vec[1], direction))
    else:
        m["head_height"] = NAN
        m["neck_elevation"] = NAN

    face_top = None
    if layout.poll and np.all(np.isfinite(P[layout.poll])):
        face_top = P[layout.poll]
    elif layout.eyes:
        eyes = np.array([P[e] for e in layout.eyes])
        eyes = eyes[np.all(np.isfinite(eyes), axis=1)]
        if len(eyes):
            face_top = eyes.mean(axis=0)
    if face_top is not None and np.all(np.isfinite(nose)):
        f = nose - face_top
        m["face_inclination"] = math.degrees(math.atan2(f[0] * direction, f[1]))
    else:
        m["face_inclination"] = NAN
    return m


def compute_metrics(xy: np.ndarray, layout: KeypointLayout) -> pd.DataFrame:
    """Metrics for all frames of a (T, K, 2) track."""
    direction = estimate_direction(xy, layout)
    rows = [frame_metrics(xy[t], layout, float(direction[t])) for t in range(len(xy))]
    df = pd.DataFrame(rows)
    df.insert(0, "direction", direction)
    return df
