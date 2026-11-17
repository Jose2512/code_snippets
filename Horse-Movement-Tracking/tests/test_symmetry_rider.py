import numpy as np
import pandas as pd
import pytest

from horsetrack.layouts import AP10K
from horsetrack.rider import _rising_diagonal, analyze_rider, rider_frame_metrics, visible_side
from horsetrack.symmetry import analyze_symmetry, stride_asymmetry

FPS, PERIOD = 60, 45  # 1.33 Hz trot


def _dips(T, depth1, depth2):
    """Up-positive vertical signal (body lengths) with one dip per half stride."""
    z = np.zeros(T)
    for s0 in range(0, T, PERIOD):
        for k, d in ((0, depth1), (1, depth2)):
            c = s0 + PERIOD * (0.25 + 0.5 * k)
            z -= d * np.exp(-((np.arange(T) - c) ** 2) / (2 * 4.0 ** 2))
    return z


def _horse_track(T, head=(0.02, 0.02), pelvis=(0.015, 0.015), bl=400.0):
    xy = np.full((T, AP10K.num_keypoints, 2), np.nan)
    neck, tail = AP10K.idx("Neck"), AP10K.idx("Root_of_tail")
    xy[:, neck] = [500, 300]
    xy[:, tail] = [100, 300]
    for e in AP10K.eyes:
        xy[:, AP10K.idx(e), 0] = 600
        xy[:, AP10K.idx(e), 1] = 200 - _dips(T, *head) * bl
    xy[:, tail, 1] = 300 - _dips(T, *pelvis) * bl
    return xy


def _gait(T):
    lf = list(range(0, T, PERIOD))
    rf = [s + PERIOD // 2 for s in lf if s + PERIOD // 2 < T]
    return {"stride_period_s": PERIOD / FPS, "gait_estimate": "trot",
            "limbs": {"LF": {"hoof_strike_frames": lf}, "RF": {"hoof_strike_frames": rf}}}


def test_head_asymmetry_points_to_lame_forelimb():
    T = 8 * PERIOD
    xy = _horse_track(T, head=(0.01, 0.03))  # head drops less in the LF half
    metrics = pd.DataFrame({"body_length_px": np.full(T, 400.0), "time_s": np.arange(T) / FPS})
    res = analyze_symmetry(xy, AP10K, metrics, _gait(T), FPS, body_length_cm=150)
    h = res["head"]
    assert res["valid_for_lameness"]
    assert h["less_drop_during_stance_of"] == "LF"
    assert h["dmin_pct_bl"] == pytest.approx(2.0, abs=0.3)
    assert h["dmin_mm"] == pytest.approx(30, abs=5)
    assert h["consistency_pct"] == 100 and h["above_literature_reference"]
    assert abs(res["pelvis"]["dmin_pct_bl"]) < 0.2  # pelvis symmetric


def test_pelvis_asymmetry_points_to_lame_hindlimb():
    T = 8 * PERIOD
    xy = _horse_track(T, pelvis=(0.02, 0.005))  # pelvis drops less in half 2 (LH stance)
    metrics = pd.DataFrame({"body_length_px": np.full(T, 400.0), "time_s": np.arange(T) / FPS})
    res = analyze_symmetry(xy, AP10K, metrics, _gait(T), FPS)
    assert res["pelvis"]["less_drop_during_stance_of"] == "LH"
    assert "dmin_mm" not in res["pelvis"]


def test_symmetry_needs_strides():
    res = analyze_symmetry(np.zeros((10, 17, 2)), AP10K, pd.DataFrame({"body_length_px": [1.0] * 10}),
                           {"stride_period_s": None}, FPS)
    assert not res["available"]
    assert stride_asymmetry(np.zeros(10), [0], [5], 10).empty


def _rider(direction=1.0, lean=0.0, heel=0.0):
    p = np.full((17, 2), np.nan)
    hip = np.array([300.0, 200.0])
    p[11] = hip
    p[5] = hip + [60 * np.sin(np.radians(lean)) * direction, -60 * np.cos(np.radians(lean))]
    p[3] = p[5] + [0, -25]
    p[13] = hip + [30 * direction, 50]
    p[15] = hip + [heel * 60 * direction, 100]
    p[7] = p[5] + [10 * direction, 35]
    p[9] = p[7] + [30 * direction, 0]
    return p


@pytest.mark.parametrize("direction", [1.0, -1.0])
def test_rider_metrics_relative_to_travel(direction):
    m = rider_frame_metrics(_rider(direction, lean=20, heel=0.5), "l", direction,
                            np.array([350.0, 260.0]), 400.0)
    assert m["trunk_lean"] == pytest.approx(20, abs=0.1)
    assert m["heel_offset"] == pytest.approx(0.5, abs=0.01)
    assert m["seat_height"] == pytest.approx(60 / 400)
    assert 90 < m["knee_angle"] < 180
    upright = rider_frame_metrics(_rider(direction), "l", direction, np.array([350.0, 260.0]), 400.0)
    assert upright["trunk_lean"] == pytest.approx(0, abs=0.1)
    assert upright["alignment_deviation"] == pytest.approx(0, abs=0.01)


def test_visible_side():
    conf = np.zeros((5, 17))
    conf[:, [4, 6, 8, 10, 12, 14, 16]] = 0.9
    assert visible_side(conf) == "r"


def test_rising_trot_diagonal_and_summary():
    T = 6 * PERIOD
    t = np.arange(T)
    lf = 20 * np.sin(2 * np.pi * t / PERIOD)          # LF retracting (stance) where its derivative < 0
    rf = -lf
    # rider sits (seat lowest) in the middle of LF stance, once per stride
    seat = -0.05 * np.cos(2 * np.pi * (t - PERIOD / 2) / PERIOD)
    hm = pd.DataFrame({"protraction_LF": lf, "protraction_RF": rf, "rider_present": True})
    d = _rising_diagonal(seat, hm, PERIOD)
    assert d["sits_with"] == "LF" and d["consistency_pct"] == 100

    rm = pd.DataFrame({f"rider_{k}": np.full(T, v) for k, v in {
        "trunk_lean": 5.0, "ear_offset": 0.0, "shoulder_offset": 0.0, "heel_offset": 0.4,
        "alignment_deviation": 0.13, "hip_angle": 120.0, "knee_angle": 120.0, "elbow_angle": 100.0,
        "hand_height": 0.5, "hand_x_horse": 0.1, "hand_y_horse": 0.2}.items()})
    rm["rider_seat_height"] = 0.2 + seat
    g = {"gait_estimate": "trot", "stride_period_s": PERIOD / FPS}
    s = analyze_rider(rm, hm, g, FPS, "l")
    assert s["trot_style"] == "rising"
    assert s["diagonal"]["sits_with"] == "LF"
    assert "chair" in s["leg_position"]
    assert s["hand_y_variability_pct_bl"] == pytest.approx(0, abs=1e-9)
