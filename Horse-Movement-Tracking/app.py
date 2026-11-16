"""Web UI: ``streamlit run app.py``

Upload a horse clip (ridden or in hand / at liberty), run the tracker and
browse the annotated video, charts and downloadable data.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import streamlit as st

from horsetrack.pipeline import Config, run
from horsetrack.pose import BACKENDS

st.set_page_config(page_title="Horse Movement Tracker", page_icon="🐎", layout="wide")
st.title("🐎 Horse Movement Tracker")
st.caption("Tracks leg joints, hooves and head position from a side-view video — with or without a rider.")

with st.sidebar:
    st.header("Models")
    backend = st.selectbox("Pose backend", BACKENDS, index=0,
                           help="vitpose = ONNX AP-10K model (recommended); yolo = your trained YOLO-pose; mmpose = RTMPose")
    pose_model = st.text_input("Pose model path", "models/vitpose-b-ap10k.onnx")
    layout = st.text_input("Keypoint layout", "ap10k", help="'ap10k' or path to a layout JSON")
    detector = st.text_input("Detector weights", "yolo11n-seg.pt")
    device = st.text_input("Device", "cpu")
    st.header("Settings")
    kpt_thr = st.slider("Keypoint confidence threshold", 0.05, 0.9, 0.3, 0.05)
    every = st.number_input("Analyse every N-th frame", 1, 10, 1)
    max_frames = st.number_input("Max frames (0 = all)", 0, 100000, 0)
    horse_id = st.number_input("Horse track ID (0 = auto)", 0, 1000, 0)
    mask_rider = st.checkbox("Paint out rider before pose estimation", False)

uploaded = st.file_uploader("Horse video", type=["mp4", "mov", "avi", "mkv", "m4v"])

if uploaded and st.button("Analyse", type="primary"):
    work = Path(tempfile.mkdtemp(prefix="horsetrack_"))
    video = work / uploaded.name
    video.write_bytes(uploaded.getbuffer())
    cfg = Config(video=str(video), output_dir=str(work / "results"), detector=detector,
                 pose_backend=backend, pose_model=pose_model or None, layout=layout or None,
                 device=device or None, kpt_thr=kpt_thr, every=int(every),
                 max_frames=int(max_frames) or None, horse_id=int(horse_id) or None, mask_rider=mask_rider)
    bar = st.progress(0.0, text="Starting…")

    def progress(stage: str, i: int, total: int) -> None:
        bar.progress(min(i / total, 1.0) if total else 0.0, text=f"{stage.capitalize()} {i}/{total or '?'}")

    try:
        st.session_state["result"] = run(cfg, progress)
    except Exception as e:  # show the error in the UI
        st.error(f"Analysis failed: {e}")
    bar.empty()

res = st.session_state.get("result")
if res:
    s, g = res.summary, res.summary["gait"]
    c = st.columns(5)
    c[0].metric("Horse detected", f"{s['horse_detected_pct']:.0f}%")
    c[1].metric("Rider", "yes" if s["ridden"] else "no", f"{s['rider_present_pct']:.0f}% of frames", delta_color="off")
    c[2].metric("Gait (estimate)", g["gait_estimate"], g["gait_confidence"], delta_color="off")
    c[3].metric("Stride frequency", f"{g['stride_frequency_hz']:.2f} Hz" if g["stride_frequency_hz"] else "–")
    hb = g["head"].get("head_bob_amplitude")
    c[4].metric("Head bob", f"{hb:.2f} BL" if hb is not None else "–", help="nose vertical range, body lengths")

    left, right = st.columns([3, 2])
    with left:
        if res.video_path and res.video_path.exists():
            st.video(str(res.video_path))
    with right:
        st.subheader("Joint range of motion (°)")
        rom = {k: v for k, v in g["joint_rom_deg"].items() if v}
        if rom:
            st.dataframe({k: {kk: round(vv, 1) for kk, vv in v.items()} for k, v in rom.items()})
        if g["left_right_rom_asymmetry_pct"]:
            st.subheader("Left/right asymmetry (%)")
            st.json({k: round(v, 1) for k, v in g["left_right_rom_asymmetry_pct"].items() if v is not None})

    for p in res.plots:
        st.image(str(p), width="stretch")

    d = st.columns(4)
    d[0].download_button("metrics.csv", (res.output_dir / "metrics.csv").read_bytes(), "metrics.csv")
    d[1].download_button("keypoints.csv", (res.output_dir / "keypoints.csv").read_bytes(), "keypoints.csv")
    d[2].download_button("summary.json", json.dumps(s, indent=2, default=str), "summary.json")
    if res.video_path:
        d[3].download_button("annotated.mp4", res.video_path.read_bytes(), "annotated.mp4")
