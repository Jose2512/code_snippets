"""End-to-end processing of a horse video.

Pass 1  detect + track the horse, find a rider, estimate keypoints.
Post    reject rider-occluded points, fill short gaps, smooth, compute
        kinematics and the clip-level gait summary.
Pass 2  render the annotated video.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import pandas as pd

from . import gait
from .detection import HorseDetector, TargetSelector, find_rider, smooth_flags
from .kinematics import compute_metrics
from .layouts import KeypointLayout, load_layout
from .pose import PoseBackend, create_pose_backend
from .report import make_plots
from .smoothing import smooth_keypoints
from .visualize import Renderer

ProgressFn = Callable[[str, int, int], None]


@dataclass
class Config:
    video: str
    output_dir: str = "outputs"
    detector: str = "yolo11n-seg.pt"
    pose_backend: str = "vitpose"
    pose_model: str | None = None
    mmpose_config: str | None = None
    layout: str | None = None  # "ap10k" or path to a layout JSON
    device: str | None = None
    det_conf: float = 0.25
    kpt_thr: float = 0.3
    rider_kpt_thr: float = 0.6  # limb points under the rider need this confidence
    horse_id: int | None = None
    every: int = 1  # analyse every N-th frame
    max_frames: int | None = None
    mask_rider: bool = False  # paint out the rider before pose estimation
    max_gap_s: float = 0.25
    smooth_window_s: float = 0.15
    render: bool = True


@dataclass
class Result:
    output_dir: Path
    summary: dict
    metrics: pd.DataFrame
    video_path: Path | None
    plots: list[Path]


def _readable(path: str) -> bool:
    cap = cv2.VideoCapture(path)
    ok = cap.isOpened() and cap.read()[0]
    cap.release()
    return bool(ok)


def prepare_video(path: str, work_dir: Path) -> str:
    """Return a path OpenCV can decode.

    iPhone clips (.MOV, HEVC/H.265, often HDR or Dolby Vision) are frequently
    not decodable by the OpenCV wheels; they are converted to H.264 with
    ffmpeg when available.
    """
    p = Path(path).expanduser()
    if not p.is_file():
        hint = ""
        if "/Videos/" in str(p):
            hint = " (on macOS the default video folder is ~/Movies)"
        raise FileNotFoundError(f"Video file not found: {p}{hint}")
    if _readable(str(p)):
        return str(p)
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError(
            f"OpenCV cannot decode {p.name} (iPhone videos are usually HEVC/H.265). "
            "Install ffmpeg (macOS: `brew install ffmpeg`) and run again, or convert it to H.264 MP4 first."
        )
    work_dir.mkdir(parents=True, exist_ok=True)
    converted = work_dir / f"{p.stem}_h264.mp4"
    print(f"Converting {p.name} to H.264 so OpenCV can read it -> {converted}", file=sys.stderr)
    r = subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-i", str(p), "-map", "0:v:0", "-an",
                        "-c:v", "libx264", "-preset", "fast", "-crf", "18", "-pix_fmt", "yuv420p",
                        str(converted)])
    if r.returncode != 0 or not _readable(str(converted)):
        raise RuntimeError(f"Could not convert {p} with ffmpeg.")
    return str(converted)


def _video_info(path: str) -> tuple[float, int, int, int]:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return fps, n, w, h


def _frames(path: str, every: int, max_frames: int | None):
    cap = cv2.VideoCapture(path)
    i = used = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if i % every == 0:
                yield i, frame
                used += 1
                if max_frames and used >= max_frames:
                    break
            i += 1
    finally:
        cap.release()


def _occluded_by_rider(xy: np.ndarray, conf: np.ndarray, rider_mask: np.ndarray,
                       layout: KeypointLayout, thr: float) -> np.ndarray:
    """Limb keypoints that lie on the rider silhouette with low confidence are
    most likely the rider's legs or guesses behind them."""
    h, w = rider_mask.shape
    occ = np.zeros(len(xy), bool)
    for i, name in enumerate(layout.keypoints):
        if layout.part_of(name) == "axial" or not np.all(np.isfinite(xy[i])):
            continue
        x, y = int(xy[i, 0]), int(xy[i, 1])
        if 0 <= x < w and 0 <= y < h and rider_mask[y, x] and conf[i] < thr:
            occ[i] = True
    return occ


def run(cfg: Config, progress: ProgressFn | None = None,
        detector: HorseDetector | None = None, pose: PoseBackend | None = None) -> Result:
    progress = progress or (lambda *_: None)
    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    layout = pose.layout if pose is not None else load_layout(cfg.layout)

    cfg = replace(cfg, video=prepare_video(cfg.video, out))
    fps_src, n_src, W, H = _video_info(cfg.video)
    fps = fps_src / cfg.every
    total = (n_src + cfg.every - 1) // cfg.every if n_src > 0 else 0
    if cfg.max_frames:
        total = min(total, cfg.max_frames) if total else cfg.max_frames

    detector = detector or HorseDetector(cfg.detector, device=cfg.device, conf=cfg.det_conf)
    pose = pose or create_pose_backend(cfg.pose_backend, layout, cfg.pose_model, cfg.device, cfg.mmpose_config)
    selector = TargetSelector(cfg.horse_id)

    K = layout.num_keypoints
    frame_idx, raw_xy, raw_conf, occluded = [], [], [], []
    horse_boxes, rider_boxes, track_ids = [], [], []
    nan4 = np.full(4, np.nan, np.float32)

    # ---------------------------------------------------------------- pass 1
    for n, (i, frame) in enumerate(_frames(cfg.video, cfg.every, cfg.max_frames)):
        dets = detector(frame)
        horse = selector.update(dets.horses)
        xy, cf, occ = np.full((K, 2), np.nan, np.float32), np.zeros(K, np.float32), np.zeros(K, bool)
        rider = None
        if horse is not None:
            rider = find_rider(horse, dets.persons)
            pose_frame = frame
            rider_mask = rider.mask(frame.shape[:2]) if rider is not None else None
            if rider is not None and cfg.mask_rider:
                pose_frame = frame.copy()
                x1, y1, x2, y2 = horse.box.astype(int)
                fill = cv2.mean(frame[max(y1, 0):y2, max(x1, 0):x2])[:3]
                pose_frame[rider_mask] = fill
            xy, cf = pose.estimate(pose_frame, horse.box)
            if rider_mask is not None:
                occ = _occluded_by_rider(xy, cf, rider_mask, layout, cfg.rider_kpt_thr)
        frame_idx.append(i)
        raw_xy.append(xy)
        raw_conf.append(cf)
        occluded.append(occ)
        horse_boxes.append(horse.box if horse is not None else nan4)
        rider_boxes.append(rider.box if rider is not None else nan4)
        track_ids.append(horse.track_id if horse is not None and horse.track_id is not None else -1)
        progress("analyzing", n + 1, total)

    if not frame_idx:
        raise RuntimeError("No frames could be read from the video.")

    raw_xy = np.stack(raw_xy)
    raw_conf = np.stack(raw_conf)
    occluded = np.stack(occluded)
    horse_boxes, rider_boxes = np.stack(horse_boxes), np.stack(rider_boxes)

    # ------------------------------------------------------------ post-process
    valid = (raw_conf >= cfg.kpt_thr) & ~occluded & np.isfinite(raw_xy[..., 0])
    xy_valid = np.where(valid[..., None], raw_xy, np.nan)
    xy_s, interpolated = smooth_keypoints(xy_valid, fps, cfg.max_gap_s, cfg.smooth_window_s)

    metrics = compute_metrics(xy_s, layout)
    times = np.asarray(frame_idx) / fps_src
    horse_detected = np.isfinite(horse_boxes[:, 0])
    rider_present = smooth_flags(np.isfinite(rider_boxes[:, 0]), int(round(0.5 * fps)))
    rider_boxes[~rider_present] = np.nan
    metrics.insert(0, "frame", frame_idx)
    metrics.insert(1, "time_s", np.round(times, 4))
    metrics.insert(2, "horse_detected", horse_detected)
    metrics.insert(3, "rider_present", rider_present)
    metrics.insert(4, "horse_track_id", track_ids)
    metrics.to_csv(out / "metrics.csv", index=False)

    _write_keypoints_csv(out / "keypoints.csv", frame_idx, times, layout, raw_xy, raw_conf,
                         xy_s, valid, interpolated, occluded)

    summary = {
        "video": {"path": str(cfg.video), "fps": fps_src, "analysed_fps": fps, "width": W, "height": H,
                  "frames_analysed": len(frame_idx), "duration_s": float(times[-1] - times[0]) if len(times) > 1 else 0.0},
        "layout": layout.name,
        "horse_detected_pct": float(100 * horse_detected.mean()),
        "rider_present_pct": float(100 * rider_present.mean()),
        "ridden": bool(rider_present.mean() > 0.3),
        "keypoint_visibility_pct": {name: float(100 * valid[:, k].mean()) for k, name in enumerate(layout.keypoints)},
        "occluded_by_rider_pct": float(100 * occluded.mean()),
        "gait": gait.analyze(metrics, fps),
        "config": asdict(cfg),
        "notes": ["All angles are 2-D projections; use a lateral (side-on) view for meaningful values.",
                  "The gait label is a heuristic estimate, not a veterinary assessment."],
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=_json_default), encoding="utf-8")
    plots = make_plots(metrics, out / "plots")

    # ---------------------------------------------------------------- pass 2
    video_path = None
    if cfg.render:
        video_path = _render(cfg, layout, fps, (W, H), frame_idx, xy_s, valid, metrics,
                             horse_boxes, rider_boxes, summary, out, progress)
    return Result(out, summary, metrics, video_path, plots)


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def _write_keypoints_csv(path, frame_idx, times, layout, raw_xy, raw_conf, xy_s, valid, interpolated, occluded):
    T, K = raw_conf.shape
    status = np.where(valid, "detected",
                      np.where(occluded, "occluded_by_rider",
                               np.where(interpolated, "interpolated", "missing")))
    df = pd.DataFrame({
        "frame": np.repeat(frame_idx, K),
        "time_s": np.round(np.repeat(times, K), 4),
        "keypoint": np.tile(layout.keypoints, T),
        "x_raw": raw_xy[..., 0].ravel(),
        "y_raw": raw_xy[..., 1].ravel(),
        "confidence": raw_conf.ravel(),
        "x": xy_s[..., 0].ravel(),
        "y": xy_s[..., 1].ravel(),
        "status": status.ravel(),
    })
    df.to_csv(path, index=False, float_format="%.2f")


def _fmt(v, spec=".0f", unit=""):
    return f"{v:{spec}}{unit}" if v is not None and np.isfinite(v) else "-"


def _render(cfg, layout, fps, size, frame_idx, xy_s, valid, metrics, horse_boxes, rider_boxes,
            summary, out, progress) -> Path:
    raw_path = out / "annotated_raw.mp4"
    writer = cv2.VideoWriter(str(raw_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    renderer = Renderer(layout, fps, frame_height=size[1])
    g = summary["gait"]
    gait_txt = f"{g['gait_estimate']} ({g['gait_confidence']})"
    freq_txt = _fmt(g["stride_frequency_hz"], ".2f", " Hz")
    rows = metrics.to_dict("records")
    for n, (i, frame) in enumerate(_frames(cfg.video, cfg.every, cfg.max_frames)):
        if n >= len(rows):
            break
        m = rows[n]
        hud = {
            "t": f"{m['time_s']:.2f}s  frame {i}",
            "rider": "yes" if m["rider_present"] else "no",
            "head height": _fmt(m.get("head_height"), "+.2f", " BL"),
            "neck elev.": _fmt(m.get("neck_elevation"), "+.0f", " deg"),
            "face vs vert.": _fmt(m.get("face_inclination"), "+.0f", " deg"),
            "gait (clip)": gait_txt,
            "stride freq": freq_txt,
        }
        if not m["horse_detected"]:
            hud["status"] = "horse not detected"
        img = renderer.draw(frame, xy_s[n], valid[n], m, horse_boxes[n], rider_boxes[n], hud)
        writer.write(img)
        progress("rendering", n + 1, len(rows))
    writer.release()

    final = out / "annotated.mp4"
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:  # re-encode to H.264 so browsers can play it
        r = subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-i", str(raw_path), "-c:v", "libx264",
                            "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(final)])
        if r.returncode == 0:
            raw_path.unlink(missing_ok=True)
            return final
    raw_path.replace(final)
    return final
