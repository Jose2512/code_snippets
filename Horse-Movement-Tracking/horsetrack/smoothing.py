"""Temporal clean-up of keypoint tracks: short-gap interpolation + smoothing."""
from __future__ import annotations

import numpy as np
from scipy.signal import savgol_filter


def _runs(valid: np.ndarray) -> list[tuple[int, int]]:
    """Contiguous True runs as [start, end) pairs."""
    runs, start = [], None
    for i, v in enumerate(valid):
        if v and start is None:
            start = i
        elif not v and start is not None:
            runs.append((start, i))
            start = None
    if start is not None:
        runs.append((start, len(valid)))
    return runs


def fill_gaps(x: np.ndarray, max_gap: int) -> np.ndarray:
    """Linearly interpolate NaN gaps of at most ``max_gap`` samples that are
    bounded by valid samples on both sides. Longer gaps stay NaN."""
    x = np.asarray(x, dtype=np.float64).copy()
    valid = ~np.isnan(x)
    for s, e in _runs(~valid):
        if s == 0 or e == len(x) or (e - s) > max_gap:
            continue
        x[s:e] = np.interp(np.arange(s, e), [s - 1, e], [x[s - 1], x[e]])
    return x


def savgol_nan(x: np.ndarray, window: int, poly: int = 2) -> np.ndarray:
    """Savitzky-Golay smoothing applied independently to each valid run."""
    x = np.asarray(x, dtype=np.float64).copy()
    if window < poly + 2:
        return x
    window = window if window % 2 == 1 else window + 1
    for s, e in _runs(~np.isnan(x)):
        if e - s >= window:
            x[s:e] = savgol_filter(x[s:e], window, poly)
    return x


def smooth_keypoints(xy: np.ndarray, fps: float, max_gap_s: float = 0.25,
                     window_s: float = 0.15) -> tuple[np.ndarray, np.ndarray]:
    """Clean a (T, K, 2) track with NaNs for missing points.

    Returns the smoothed track and a (T, K) bool array marking the samples
    that were filled by interpolation.
    """
    T, K, _ = xy.shape
    max_gap = max(1, int(round(max_gap_s * fps)))
    window = max(5, int(round(window_s * fps)) | 1)
    out = np.full_like(xy, np.nan, dtype=np.float64)
    missing = np.isnan(xy[..., 0])
    for k in range(K):
        for d in range(2):
            out[:, k, d] = savgol_nan(fill_gaps(xy[:, k, d], max_gap), window)
    interpolated = missing & ~np.isnan(out[..., 0])
    return out, interpolated
