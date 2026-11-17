"""Rider position and technique from a human pose model (COCO 17 keypoints).

Measured on the rider's side facing the camera (the far side is hidden by
the horse). All values are 2-D, expressed relative to the horse's direction
of travel (forward = toward the horse's head):

* ``trunk_lean``: angle of hip->shoulder from vertical (+ forward, - behind).
* ``ear/shoulder/heel_offset``: horizontal distance of each point from the
  vertical line through the hip, in trunk lengths (+ ahead). The classical
  ear-shoulder-hip-heel alignment has all three close to 0. A heel well
  ahead of the hip suggests a "chair seat"; a heel behind the hip with the
  trunk tipped forward suggests a "fork seat".
* ``hip_angle`` (shoulder-hip-knee), ``knee_angle`` (hip-knee-ankle),
  ``elbow_angle`` (shoulder-elbow-wrist), degrees.
* ``hand_height``: wrist height above the hip, in trunk lengths.
* ``seat_height``: rider hip height above the horse's withers point, in
  horse body lengths. Its oscillation shows rising vs sitting trot.
* ``hand_x_horse / hand_y_horse``: wrist relative to the withers point, in
  horse body lengths; their variability measures how independent and steady
  the hands are.

Educational feedback for training; it does not replace an instructor.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .detection import box_iou
from .kinematics import joint_angle

COCO_KPTS = ("nose", "l_eye", "r_eye", "l_ear", "r_ear", "l_shoulder", "r_shoulder", "l_elbow", "r_elbow",
             "l_wrist", "r_wrist", "l_hip", "r_hip", "l_knee", "r_knee", "l_ankle", "r_ankle")
SIDE = {
    "l": dict(ear=3, shoulder=5, elbow=7, wrist=9, hip=11, knee=13, ankle=15),
    "r": dict(ear=4, shoulder=6, elbow=8, wrist=10, hip=12, knee=14, ankle=16),
}
CHAIN = ("ear", "shoulder", "elbow", "wrist", "hip", "knee", "ankle")
NAN = float("nan")


class RiderPose:
    """Ultralytics YOLO-pose (person) restricted to the rider box."""

    def __init__(self, weights: str = "yolo11n-pose.pt", device: str | None = None) -> None:
        from ultralytics import YOLO

        self.model = YOLO(weights)
        self.device = device

    def estimate(self, frame: np.ndarray, rider_box: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        empty = np.full((17, 2), np.nan, np.float32), np.zeros(17, np.float32)
        r = self.model(frame, classes=[0], conf=0.2, device=self.device, verbose=False)[0]
        if r.keypoints is None or r.boxes is None or len(r.boxes) == 0:
            return empty
        boxes = r.boxes.xyxy.cpu().numpy()
        ious = [box_iou(b, rider_box) for b in boxes]
        i = int(np.argmax(ious))
        if ious[i] < 0.3:
            return empty
        xy = r.keypoints.xy[i].cpu().numpy().astype(np.float32)
        conf = r.keypoints.conf[i].cpu().numpy().astype(np.float32) if r.keypoints.conf is not None \
            else np.ones(17, np.float32)
        xy[conf <= 0] = np.nan
        return xy, conf


def visible_side(conf: np.ndarray) -> str:
    """Side of the rider facing the camera, from mean keypoint confidence."""
    cl = np.nanmean([conf[:, SIDE["l"][k]] for k in CHAIN])
    cr = np.nanmean([conf[:, SIDE["r"][k]] for k in CHAIN])
    return "l" if cl >= cr else "r"


def rider_frame_metrics(p: np.ndarray, side: str, direction: float, withers: np.ndarray,
                        horse_body_len: float) -> dict[str, float]:
    J = {k: p[i] for k, i in SIDE[side].items()}
    m: dict[str, float] = {}
    hip, sho = J["hip"], J["shoulder"]
    trunk = sho - hip
    tl = float(np.linalg.norm(trunk)) if np.all(np.isfinite(trunk)) else NAN
    m["trunk_lean"] = math.degrees(math.atan2(trunk[0] * direction, -trunk[1])) if np.isfinite(tl) else NAN
    for k in ("ear", "shoulder", "ankle"):
        name = "heel_offset" if k == "ankle" else f"{k}_offset"
        d = J[k] - hip
        m[name] = float(d[0] * direction / tl) if np.isfinite(tl) and tl > 0 and np.all(np.isfinite(d)) else NAN
    offs = [m["ear_offset"], m["shoulder_offset"], m["heel_offset"]]
    m["alignment_deviation"] = float(np.mean(np.abs(offs))) if all(np.isfinite(offs)) else NAN
    m["hip_angle"] = joint_angle(J["shoulder"], J["hip"], J["knee"])
    m["knee_angle"] = joint_angle(J["hip"], J["knee"], J["ankle"])
    m["elbow_angle"] = joint_angle(J["shoulder"], J["elbow"], J["wrist"])
    m["hand_height"] = float((hip[1] - J["wrist"][1]) / tl) if np.isfinite(tl) and tl > 0 else NAN
    ok_h = np.isfinite(horse_body_len) and horse_body_len > 0 and np.all(np.isfinite(withers))
    m["seat_height"] = float((withers[1] - hip[1]) / horse_body_len) if ok_h and np.all(np.isfinite(hip)) else NAN
    w = J["wrist"]
    m["hand_x_horse"] = float((w[0] - withers[0]) * direction / horse_body_len) if ok_h and np.all(np.isfinite(w)) else NAN
    m["hand_y_horse"] = float((withers[1] - w[1]) / horse_body_len) if ok_h and np.all(np.isfinite(w)) else NAN
    return m


def compute_rider_metrics(rider_xy: np.ndarray, rider_conf: np.ndarray, horse_xy: np.ndarray,
                          layout, direction: np.ndarray, body_len: np.ndarray) -> tuple[pd.DataFrame, str]:
    side = visible_side(rider_conf)
    wi = layout.idx(layout.withers_point)
    bl = float(np.nanmedian(body_len)) if np.isfinite(body_len).any() else NAN
    rows = [rider_frame_metrics(rider_xy[t], side, float(direction[t]), horse_xy[t, wi], bl)
            for t in range(len(rider_xy))]
    return pd.DataFrame(rows).add_prefix("rider_"), side


def _stats(x: pd.Series) -> dict | None:
    x = x.dropna()
    if len(x) < 5:
        return None
    return {"mean": float(x.mean()), "sd": float(x.std()), "p5": float(x.quantile(0.05)),
            "p95": float(x.quantile(0.95))}


def _dominant_period(x: np.ndarray, fps: float) -> float | None:
    from .gait import dominant_period

    return dominant_period(x, fps, fmin=0.5, fmax=6.0)


def analyze_rider(rm: pd.DataFrame, horse_metrics: pd.DataFrame, gait_summary: dict, fps: float,
                  side: str, body_length_cm: float | None = None) -> dict:
    present = horse_metrics["rider_present"].to_numpy(bool)
    valid = rm["rider_trunk_lean"].notna().to_numpy() & present
    if valid.sum() < max(10, int(fps)):
        return {"available": False, "reason": "rider not visible long enough"}
    r = rm[valid]
    out: dict = {"available": True, "frames": int(valid.sum()), "side_measured": "left" if side == "l" else "right"}
    for col in ("trunk_lean", "ear_offset", "shoulder_offset", "heel_offset", "alignment_deviation",
                "hip_angle", "knee_angle", "elbow_angle", "hand_height"):
        out[col] = _stats(r[f"rider_{col}"])

    tl = r["rider_trunk_lean"]
    out["trunk_forward_gt_15deg_pct"] = float(100 * (tl > 15).mean())
    out["trunk_behind_gt_10deg_pct"] = float(100 * (tl < -10).mean())
    heel = out["heel_offset"]["mean"] if out["heel_offset"] else None
    if heel is not None:
        out["leg_position"] = ("heel ahead of hip (chair-seat tendency)" if heel > 0.25 else
                               "heel behind hip (fork-seat tendency)" if heel < -0.25 else
                               "heel roughly under hip")

    # hand steadiness relative to the horse (high-pass over one stride)
    period_s = gait_summary.get("stride_period_s")
    win = max(3, int(round((period_s or 1.0) * fps)))
    mm = body_length_cm * 10 if body_length_cm else None
    for axis in ("x", "y"):
        s = rm[f"rider_hand_{axis}_horse"].where(pd.Series(present))
        hp = s - s.rolling(win, center=True, min_periods=1).mean()
        sd = float(hp.std()) if hp.notna().sum() > 5 else None
        out[f"hand_{axis}_variability_pct_bl"] = 100 * sd if sd is not None else None
        if mm and sd is not None:
            out[f"hand_{axis}_variability_mm"] = sd * mm

    # rising vs sitting trot and diagonal
    seat = rm["rider_seat_height"].where(pd.Series(present)).to_numpy(float)
    out["seat_movement_pct_bl"] = None
    if np.isfinite(seat).sum() > 2 * fps:
        hp = seat - pd.Series(seat).rolling(win, center=True, min_periods=1).mean().to_numpy()
        out["seat_movement_pct_bl"] = float(100 * (np.nanpercentile(hp, 95) - np.nanpercentile(hp, 5)))
        sp = _dominant_period(seat, fps)
        if gait_summary.get("gait_estimate") == "trot" and period_s and sp:
            ratio = sp / (period_s * fps)
            out["trot_style"] = "rising" if 0.75 < ratio < 1.3 else "sitting / two-point" if 0.35 < ratio < 0.65 else "unclear"
            if out["trot_style"] == "rising":
                out["diagonal"] = _rising_diagonal(hp, horse_metrics, period_s * fps)
    out["notes"] = [
        "2-D measurement of the rider side facing the camera; the far leg is hidden by the horse.",
        "Rising trot: by convention the rider sits when the OUTSIDE foreleg is on the ground.",
        "Educational feedback for training; validate with a qualified instructor.",
    ]
    return out


def _rising_diagonal(seat_hp: np.ndarray, hm: pd.DataFrame, period: float) -> dict:
    """Which foreleg is in stance when the rider sits (seat lowest)."""
    from scipy.signal import find_peaks

    y = np.nan_to_num(-seat_hp, nan=np.nanmin(-seat_hp))
    sits, _ = find_peaks(y, distance=max(1, int(0.6 * period)))
    lf = hm.get("protraction_LF")
    rf = hm.get("protraction_RF")
    if lf is None or rf is None or not len(sits):
        return {"sits_with": "undetermined"}
    vl, vr = np.gradient(lf.interpolate().to_numpy()), np.gradient(rf.interpolate().to_numpy())
    left = sum(1 for s in sits if vl[s] < 0 <= vr[s])   # LF retracting = in stance
    right = sum(1 for s in sits if vr[s] < 0 <= vl[s])
    n = left + right
    if not n:
        return {"sits_with": "undetermined"}
    lead = "LF" if left >= right else "RF"
    return {"sits_with": lead, "consistency_pct": float(100 * max(left, right) / n), "sits_counted": int(n),
            "diagonal_name": "left diagonal" if lead == "LF" else "right diagonal",
            "correct_on": "right rein (clockwise)" if lead == "LF" else "left rein (anticlockwise)"}
