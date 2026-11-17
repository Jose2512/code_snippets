"""Vertical movement asymmetry of head, withers and pelvis (lameness screening).

Method (as in published objective lameness studies, adapted to 2-D video):
a sound horse at trot moves its head, withers and pelvis down and up twice
per stride, symmetrically for the two diagonal stance phases. A horse that
unloads a painful limb shows *less downward movement* (and/or less upward
push-off) during that limb's stance.

For every stride (between two left-fore hoof strikes) the vertical signal is
split at the right-fore strike into

* half 1 — left-fore / right-hind diagonal stance
* half 2 — right-fore / left-hind diagonal stance

and we compute ``DMin = min(half 1) - min(half 2)`` and
``DMax = max(half 1) - max(half 2)`` (vertical axis pointing up).

Interpretation of DMin (median across strides):

* head: DMin > 0 -> less head drop during LF stance (points to LF);
  DMin < 0 -> RF.
* pelvis: half 1 contains the RH stance, so DMin > 0 -> points to RH;
  DMin < 0 -> LH.
* withers: helps to tell a primary forelimb lameness from a compensatory
  head nod caused by a hindlimb lameness.

Values are in % of body length (neck->tail-root distance) and in mm when the
real body length is given. Only meaningful at trot, filmed side-on on a
straight line with a fixed camera. This is educational orientation only and
does not replace a veterinary examination.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Reference values reported in the literature for inertial-sensor systems
# (mean of strides, mm). Used only to *flag* values for review; they are not
# validated for this video method.
REFERENCE_MM = {"head": 6.0, "pelvis": 3.0, "withers": 3.0}

INTERPRET = {
    "head": ("LF", "RF"),
    "withers": ("LF", "RF"),
    "pelvis": ("RH", "LH"),
}


def _interp(x: np.ndarray) -> np.ndarray | None:
    x = np.asarray(x, float)
    ok = np.isfinite(x)
    if ok.sum() < max(10, 0.5 * len(x)):
        return None
    i = np.arange(len(x))
    return np.interp(i, i[ok], x[ok])


def vertical_signal(y: np.ndarray, body_len_px: float, period: float) -> np.ndarray | None:
    """Image y (pixels, down) -> up-positive displacement in body lengths with
    the slow trend (camera tilt, terrain, framing) removed."""
    z = _interp(y)
    if z is None or not np.isfinite(body_len_px) or body_len_px <= 0:
        return None
    z = -z / body_len_px
    w = max(3, int(round(period)))
    trend = pd.Series(z).rolling(w, center=True, min_periods=1).mean().to_numpy()
    return z - trend


def stride_asymmetry(z: np.ndarray, strikes_lf: list[int], strikes_rf: list[int], period: float) -> pd.DataFrame:
    rows = []
    rf = np.asarray(strikes_rf)
    for s0, s1 in zip(strikes_lf, strikes_lf[1:]):
        if not (0.6 * period <= s1 - s0 <= 1.5 * period):
            continue
        inside = rf[(rf > s0 + 2) & (rf < s1 - 2)]
        mid = int(inside[0]) if len(inside) else (s0 + s1) // 2
        h1, h2 = z[s0:mid], z[mid:s1]
        if len(h1) < 2 or len(h2) < 2:
            continue
        rows.append({"start_frame": s0, "end_frame": s1,
                     "dmin": float(h1.min() - h2.min()), "dmax": float(h1.max() - h2.max()),
                     "range": float(z[s0:s1].max() - z[s0:s1].min())})
    return pd.DataFrame(rows)


def analyze_symmetry(xy: np.ndarray, layout, metrics: pd.DataFrame, gait_summary: dict, fps: float,
                     body_length_cm: float | None = None) -> dict:
    """xy: smoothed (T, K, 2) horse keypoints."""
    period_s = gait_summary.get("stride_period_s")
    limbs = gait_summary.get("limbs") or {}
    if not period_s or not limbs.get("LF") or not limbs.get("RF"):
        return {"available": False, "reason": "no regular stride detected"}
    period = period_s * fps
    lf, rf = limbs["LF"].get("hoof_strike_frames", []), limbs["RF"].get("hoof_strike_frames", [])
    body_len = float(np.nanmedian(metrics["body_length_px"])) if "body_length_px" in metrics else np.nan

    head_pt = layout.poll
    head_y = xy[:, layout.idx(head_pt), 1] if head_pt else None
    if head_y is None and layout.eyes:
        head_y = np.nanmean(np.stack([xy[:, layout.idx(e), 1] for e in layout.eyes]), axis=0)
    if head_y is None:
        head_y = xy[:, layout.idx(layout.nose), 1]
    signals = {
        "head": head_y,
        "withers": xy[:, layout.idx(layout.withers_point), 1],
        "pelvis": xy[:, layout.idx(layout.pelvis_point), 1],
    }
    mm_per_bl = body_length_cm * 10 if body_length_cm else None
    out: dict = {"available": True, "gait": gait_summary.get("gait_estimate"),
                 "valid_for_lameness": gait_summary.get("gait_estimate") == "trot",
                 "units": "percent of body length" + (" and mm" if mm_per_bl else ""),
                 "per_stride": {}}
    for name, y in signals.items():
        z = vertical_signal(y, body_len, period)
        if z is None:
            out[name] = None
            continue
        df = stride_asymmetry(z, lf, rf, period)
        if df.empty:
            out[name] = None
            continue
        med_min, med_max = float(df["dmin"].median()), float(df["dmax"].median())
        sign = np.sign(med_min)
        res = {
            "strides": int(len(df)),
            "dmin_pct_bl": 100 * med_min,
            "dmax_pct_bl": 100 * float(med_max),
            "dmin_iqr_pct_bl": 100 * float(df["dmin"].quantile(0.75) - df["dmin"].quantile(0.25)),
            "vertical_range_pct_bl": 100 * float(df["range"].median()),
            "consistency_pct": float(100 * (np.sign(df["dmin"]) == sign).mean()) if sign else 0.0,
            "less_drop_during_stance_of": INTERPRET[name][0] if med_min > 0 else INTERPRET[name][1],
        }
        if mm_per_bl:
            res["dmin_mm"] = med_min * mm_per_bl
            res["dmax_mm"] = med_max * mm_per_bl
            ref = REFERENCE_MM[name]
            res["above_literature_reference"] = bool(abs(res["dmin_mm"]) > ref)
            res["literature_reference_mm"] = ref
        out[name] = res
        out["per_stride"][name] = df.to_dict("records")
    out["notes"] = [
        "Positive DMin = less downward movement in the first half of the stride (LF/RH diagonal stance).",
        "A hindlimb lameness can cause a compensatory head nod; compare head, withers and pelvis together.",
        "Repeatability of published systems is a few mm; low consistency or few strides means low reliability.",
        "Educational orientation only, not a veterinary diagnosis. Confirm with a veterinarian.",
    ]
    return out
