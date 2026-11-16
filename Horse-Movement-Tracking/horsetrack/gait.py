"""Clip-level gait analysis from the per-frame limb signals.

Everything here works on the *protraction* angle of each limb, which is
independent of camera panning: during stance the limb rotates backward
(protraction decreases), during swing it swings forward.

The gait label is a heuristic based on the phase relationships between the
limbs (e.g. trot = diagonal pairs move together) and should be treated as
an estimate, not a veterinary assessment.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

LIMBS = ("LF", "RF", "LH", "RH")


def _interp_nan(x: np.ndarray) -> np.ndarray | None:
    x = np.asarray(x, float)
    ok = np.isfinite(x)
    if ok.sum() < max(10, 0.5 * len(x)):
        return None
    idx = np.arange(len(x))
    return np.interp(idx, idx[ok], x[ok])


def dominant_period(x: np.ndarray, fps: float, fmin: float = 0.4, fmax: float = 3.5) -> float | None:
    """Stride period in frames, from the first strong autocorrelation peak."""
    y = _interp_nan(x)
    if y is None:
        return None
    y = y - y.mean()
    if np.allclose(y, 0):
        return None
    ac = np.correlate(y, y, mode="full")[len(y) - 1:]
    ac /= ac[0]
    lo, hi = int(fps / fmax), min(int(fps / fmin), len(ac) - 1)
    if hi <= lo + 2:
        return None
    peaks, props = find_peaks(ac[lo:hi + 1], height=0.2)
    if not len(peaks):
        return None
    best = peaks[np.argmax(props["peak_heights"])]
    # prefer the first peak that is nearly as high (avoid picking 2x period)
    for p, h in zip(peaks, props["peak_heights"]):
        if h >= 0.8 * props["peak_heights"].max():
            best = p
            break
    k = lo + best
    # parabolic refinement
    if 0 < k < len(ac) - 1:
        a, b, c = ac[k - 1], ac[k], ac[k + 1]
        denom = a - 2 * b + c
        if denom != 0:
            return float(k + 0.5 * (a - c) / denom)
    return float(k)


def phase_lag(a: np.ndarray, b: np.ndarray, period: float) -> float | None:
    """Fraction of the stride (0..1) by which ``b`` lags ``a``."""
    ya, yb = _interp_nan(a), _interp_nan(b)
    if ya is None or yb is None:
        return None
    ya, yb = ya - ya.mean(), yb - yb.mean()
    n = len(ya)
    best, best_c = 0, -np.inf
    for lag in range(int(round(period))):
        c = np.dot(ya[: n - lag], yb[lag:]) / max(n - lag, 1)
        if c > best_c:
            best, best_c = lag, c
    return float(best / period) % 1.0


def _d0(x: float) -> float:
    """Circular distance of a phase (0..1) to 0."""
    return min(x % 1.0, 1 - x % 1.0)


def classify_gait(lags: dict[str, float | None], freq_hz: float | None) -> tuple[str, str]:
    """Return (gait, confidence)."""
    need = ("LF-RH", "RF-LH", "LF-LH", "RF-RH", "LF-RF")
    if freq_hz is None or any(lags.get(k) is None for k in need):
        return "undetermined", "none"
    diag1, diag2 = _d0(lags["LF-RH"]), _d0(lags["RF-LH"])
    lat_l, lat_r = _d0(lags["LF-LH"]), _d0(lags["RF-RH"])
    fore = _d0(lags["LF-RF"])
    tol = 0.12
    if diag1 < tol and diag2 < tol and abs(fore - 0.5) < 0.15:
        return "trot", "high" if max(diag1, diag2) < 0.07 else "medium"
    if lat_l < tol and lat_r < tol and abs(fore - 0.5) < 0.15:
        return "pace", "medium"
    if (diag1 < tol) != (diag2 < tol) and fore < 0.4 and freq_hz > 1.2:
        return "canter", "medium"
    if abs(fore - 0.5) < 0.15 and 0.12 <= diag1 <= 0.38 and 0.12 <= diag2 <= 0.38:
        return "walk", "high" if freq_hz < 1.3 else "medium"
    if fore < 0.25 and freq_hz > 1.6:
        return "gallop", "low"
    return "undetermined", "low"


def limb_events(protraction: np.ndarray, period: float, fps: float) -> dict:
    """Stance/swing estimate for one limb from its protraction signal."""
    y = _interp_nan(protraction)
    if y is None:
        return {}
    vel = np.gradient(y) * fps
    stance = vel < 0  # limb rotating backward relative to the body
    peaks, _ = find_peaks(y, distance=max(1, int(0.6 * period)), prominence=2.0)
    stride_times = np.diff(peaks) / fps if len(peaks) > 1 else np.array([])
    return {
        "duty_factor": float(stance.mean()),
        "max_protraction_deg": float(np.nanpercentile(protraction, 95)),
        "max_retraction_deg": float(np.nanpercentile(protraction, 5)),
        "swing_arc_deg": float(np.nanpercentile(protraction, 95) - np.nanpercentile(protraction, 5)),
        "hoof_strike_frames": peaks.tolist(),
        "strides_counted": int(max(len(peaks) - 1, 0)),
        "stride_time_mean_s": float(stride_times.mean()) if len(stride_times) else None,
        "stride_time_cv": float(stride_times.std() / stride_times.mean()) if len(stride_times) > 1 else None,
    }


def _rom(x: pd.Series) -> dict:
    x = x.dropna()
    if len(x) < 5:
        return {}
    p5, p95 = np.percentile(x, [5, 95])
    return {"min": float(p5), "max": float(p95), "mean": float(x.mean()), "rom": float(p95 - p5)}


def symmetry_index(left: float | None, right: float | None) -> float | None:
    """0 % = perfectly symmetric."""
    if left is None or right is None or (left + right) == 0:
        return None
    return float(200 * abs(left - right) / (left + right))


def analyze(metrics: pd.DataFrame, fps: float) -> dict:
    """Summarise a clip. ``metrics`` comes from :func:`kinematics.compute_metrics`."""
    prot = {l: metrics.get(f"protraction_{l}") for l in LIMBS}
    periods = [dominant_period(s.to_numpy(), fps) for s in prot.values() if s is not None]
    periods = [p for p in periods if p]
    period = float(np.median(periods)) if periods else None
    freq = fps / period if period else None

    lags: dict[str, float | None] = {}
    if period:
        for a, b in (("LF", "RH"), ("RF", "LH"), ("LF", "LH"), ("RF", "RH"), ("LF", "RF"), ("LH", "RH")):
            lags[f"{a}-{b}"] = phase_lag(prot[a].to_numpy(), prot[b].to_numpy(), period)
    gait, conf = classify_gait(lags, freq)

    limbs = {l: limb_events(prot[l].to_numpy(), period, fps) for l in LIMBS} if period else {}
    joints = {c[len("angle_"):]: _rom(metrics[c]) for c in metrics.columns if c.startswith("angle_")}

    sym = {}
    for name, r in joints.items():
        if name.startswith("LF_") or name.startswith("LH_"):
            other = "R" + name[1:]
            if other in joints and joints[name] and joints[other]:
                key = ("fore" if name[1] == "F" else "hind") + name[2:]  # e.g. "fore_elbow"
                sym[key] = symmetry_index(joints[name]["rom"], joints[other]["rom"])

    head = {}
    if "head_height" in metrics:
        hh = metrics["head_height"].dropna()
        if len(hh) > 5:
            p5, p95 = np.percentile(hh, [5, 95])
            head["head_height_mean"] = float(hh.mean())
            head["head_bob_amplitude"] = float(p95 - p5)  # in body lengths
    for col in ("neck_elevation", "face_inclination"):
        if col in metrics and metrics[col].notna().sum() > 5:
            head[col] = _rom(metrics[col])

    return {
        "stride_period_s": period / fps if period else None,
        "stride_frequency_hz": freq,
        "gait_estimate": gait,
        "gait_confidence": conf,
        "phase_lags": lags,
        "limbs": limbs,
        "joint_rom_deg": joints,
        "left_right_rom_asymmetry_pct": sym,
        "head": head,
    }
