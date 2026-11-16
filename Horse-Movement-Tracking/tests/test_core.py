import math

import numpy as np
import pandas as pd
import pytest

from horsetrack.detection import Detection, TargetSelector, find_rider
from horsetrack.gait import analyze, classify_gait, dominant_period, phase_lag
from horsetrack.kinematics import compute_metrics, frame_metrics, joint_angle
from horsetrack.layouts import AP10K, KeypointLayout, load_layout
from horsetrack.smoothing import fill_gaps, smooth_keypoints


def standing_pose(direction=1.0):
    """A simple side-view horse facing right (direction=+1) or left (-1)."""
    P = {
        "Neck": (300, 200), "Root_of_tail": (100, 200), "Nose": (380, 150),
        "L_Eye": (360, 120), "R_Eye": (360, 120),
        "L_Shoulder": (280, 220), "L_Elbow": (280, 300), "L_F_Paw": (280, 400),
        "R_Shoulder": (280, 220), "R_Elbow": (290, 300), "R_F_Paw": (330, 390),
        "L_Hip": (120, 220), "L_Knee": (130, 300), "L_B_Paw": (120, 400),
        "R_Hip": (120, 220), "R_Knee": (130, 300), "R_B_Paw": (120, 400),
    }
    xy = np.array([P[k] for k in AP10K.keypoints], float)
    if direction < 0:
        xy[:, 0] = 500 - xy[:, 0]
    return xy


def test_layouts_load():
    assert load_layout(None) is AP10K
    detailed = load_layout("configs/horse_detailed_layout.json")
    assert detailed.num_keypoints == 27
    assert detailed.hooves["LH"] == "LH_Hoof"
    with pytest.raises(ValueError):
        KeypointLayout.from_dict({**{"name": "bad", "keypoints": ["a"], "skeleton": [["a", "b"]],
                                     "limbs": {}, "head": {"nose": "a", "neck": "a"}, "body_axis": ["a", "a"]}})


def test_joint_angle():
    assert joint_angle(np.array([0, 1.0]), np.array([0, 0.0]), np.array([1.0, 0])) == pytest.approx(90)
    assert joint_angle(np.array([0, -1.0]), np.array([0, 0.0]), np.array([0, 1.0])) == pytest.approx(180)
    assert math.isnan(joint_angle(np.array([np.nan, 0]), np.array([0, 0.0]), np.array([1.0, 0])))


@pytest.mark.parametrize("direction", [1.0, -1.0])
def test_frame_metrics_orientation_invariant(direction):
    m = frame_metrics(standing_pose(direction), AP10K, direction)
    assert m["angle_LF_elbow"] == pytest.approx(180)
    assert m["protraction_LF"] == pytest.approx(0, abs=1e-6)
    assert m["protraction_RF"] == pytest.approx(math.degrees(math.atan2(50, 170)))  # right fore hoof is ahead of the shoulder
    assert m["head_height"] == pytest.approx(0.25)
    assert m["neck_elevation"] == pytest.approx(math.degrees(math.atan2(50, 80)))
    assert m["face_inclination"] > 0  # nose ahead of the eyes


def test_direction_estimated_from_pose():
    xy = np.stack([standing_pose(-1)] * 20)
    df = compute_metrics(xy, AP10K)
    assert (df["direction"] == -1).all()
    assert df["protraction_RF"].iloc[0] > 15


def test_fill_gaps_and_smoothing():
    x = np.array([0, 1, np.nan, np.nan, 4, np.nan, np.nan, np.nan, np.nan, 9], float)
    f = fill_gaps(x, max_gap=2)
    assert f[2] == pytest.approx(2) and f[3] == pytest.approx(3)
    assert np.isnan(f[5:9]).all()
    xy = np.random.default_rng(0).normal(0, 1, (60, 2, 2)) + np.linspace(0, 100, 60)[:, None, None]
    xy[10, 0] = np.nan
    out, interp = smooth_keypoints(xy, fps=30)
    assert interp[10, 0] and not interp[11, 0]
    assert np.isfinite(out).all()


def _limb_signals(phases, f=1.5, fps=30, secs=6):
    t = np.arange(int(fps * secs)) / fps
    return {k: 25 * np.sin(2 * np.pi * (f * t - ph)) for k, ph in phases.items()}


@pytest.mark.parametrize("gait,phases,f", [
    ("trot", {"LF": 0, "RH": 0, "RF": 0.5, "LH": 0.5}, 1.4),
    ("walk", {"LH": 0, "LF": 0.25, "RH": 0.5, "RF": 0.75}, 0.9),
    ("canter", {"RH": 0, "LH": 0.3, "RF": 0.3, "LF": 0.55}, 1.8),
    ("pace", {"LF": 0, "LH": 0, "RF": 0.5, "RH": 0.5}, 1.4),
])
def test_gait_classification(gait, phases, f):
    sig = _limb_signals(phases, f=f)
    df = pd.DataFrame({f"protraction_{k}": v for k, v in sig.items()})
    s = analyze(df, fps=30)
    assert s["gait_estimate"] == gait
    assert s["stride_frequency_hz"] == pytest.approx(f, rel=0.05)
    assert 0.3 < s["limbs"]["LF"]["duty_factor"] < 0.7


def test_period_and_lag():
    sig = _limb_signals({"a": 0, "b": 0.25}, f=1.0)
    p = dominant_period(sig["a"], 30)
    assert p == pytest.approx(30, rel=0.03)
    assert phase_lag(sig["a"], sig["b"], p) == pytest.approx(0.25, abs=0.05)
    assert classify_gait({}, None) == ("undetermined", "none")


def det(box, cls=17, tid=None, conf=0.9):
    return Detection(np.array(box, np.float32), conf, cls, tid)


def test_rider_vs_handler():
    horse = det([100, 100, 500, 400])
    rider = det([250, 20, 350, 300], cls=0)
    handler = det([420, 150, 490, 400], cls=0)  # standing on the ground beside the horse
    assert find_rider(horse, [handler]) is None
    assert find_rider(horse, [handler, rider]) is rider


def test_target_selector_follows_track():
    sel = TargetSelector()
    a, b = det([0, 0, 100, 100], tid=1), det([200, 0, 260, 50], tid=2)
    assert sel.update([a, b]).track_id == 1
    # tracker re-assigns an ID: keep following by overlap
    a2 = det([5, 0, 105, 100], tid=7)
    assert sel.update([b, a2]).track_id == 7
    assert sel.update([]) is None
    fixed = TargetSelector(horse_id=2)
    assert fixed.update([a, b]).track_id == 2


def test_smooth_flags_removes_flicker():
    from horsetrack.detection import smooth_flags

    f = np.zeros(30, bool)
    f[10] = True
    f[20:30] = True
    out = smooth_flags(f, 5)
    assert not out[10] and out[25]


def test_prepare_video_reports_missing_file(tmp_path):
    from horsetrack.pipeline import prepare_video

    with pytest.raises(FileNotFoundError, match="Movies"):
        prepare_video(str(tmp_path / "Videos" / "IMG_0001.MOV"), tmp_path)


def test_prepare_video_converts_unreadable_file(tmp_path):
    import shutil
    import subprocess

    from horsetrack.pipeline import _readable, prepare_video

    if not shutil.which("ffmpeg"):
        pytest.skip("ffmpeg not installed")
    src = tmp_path / "clip.MOV"  # HEVC in a QuickTime container, like an iPhone clip
    subprocess.run(["ffmpeg", "-loglevel", "error", "-f", "lavfi", "-i", "testsrc=size=320x240:rate=30",
                    "-t", "1", "-c:v", "libx265", "-tag:v", "hvc1", str(src)], check=True)
    out = prepare_video(str(src), tmp_path / "out")
    assert _readable(out)
