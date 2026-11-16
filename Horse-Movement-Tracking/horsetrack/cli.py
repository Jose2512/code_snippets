"""Command line interface: ``python -m horsetrack video.mp4 -o outputs/``."""
from __future__ import annotations

import argparse
import sys

from .pipeline import Config, run
from .pose import BACKENDS


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="horsetrack", description="Track horse leg joints and head position in a video.")
    p.add_argument("video", help="input video (with or without rider)")
    p.add_argument("-o", "--output-dir", default="outputs", help="folder for results (default: outputs)")
    p.add_argument("--pose-backend", choices=BACKENDS, default="vitpose")
    p.add_argument("--pose-model", help="pose weights: .onnx (vitpose), .pt (yolo) or checkpoint (mmpose)")
    p.add_argument("--mmpose-config", help="MMPose config file (mmpose backend)")
    p.add_argument("--layout", help="keypoint layout: 'ap10k' (default) or path to a layout JSON")
    p.add_argument("--detector", default="yolo11n-seg.pt", help="Ultralytics detector weights (COCO)")
    p.add_argument("--device", help="cpu, cuda, cuda:0, mps ...")
    p.add_argument("--det-conf", type=float, default=0.25, help="detection confidence threshold")
    p.add_argument("--kpt-thr", type=float, default=0.3, help="keypoint confidence threshold")
    p.add_argument("--rider-kpt-thr", type=float, default=0.6,
                   help="confidence a limb point covered by the rider needs to be kept")
    p.add_argument("--horse-id", type=int, help="follow this tracker ID (when several horses are visible)")
    p.add_argument("--every", type=int, default=1, help="analyse every N-th frame")
    p.add_argument("--max-frames", type=int, help="stop after N analysed frames")
    p.add_argument("--mask-rider", action="store_true", help="paint the rider out before pose estimation")
    p.add_argument("--no-video", action="store_true", help="skip rendering the annotated video")
    return p


def main(argv: list[str] | None = None) -> int:
    a = build_parser().parse_args(argv)
    cfg = Config(
        video=a.video, output_dir=a.output_dir, detector=a.detector, pose_backend=a.pose_backend,
        pose_model=a.pose_model, mmpose_config=a.mmpose_config, layout=a.layout, device=a.device,
        det_conf=a.det_conf, kpt_thr=a.kpt_thr, rider_kpt_thr=a.rider_kpt_thr, horse_id=a.horse_id,
        every=max(1, a.every), max_frames=a.max_frames, mask_rider=a.mask_rider, render=not a.no_video,
    )
    last = {"stage": None}

    def progress(stage: str, i: int, total: int) -> None:
        if stage != last["stage"]:
            if last["stage"]:
                sys.stderr.write("\n")
            last["stage"] = stage
        if i % 10 == 0 or i == total:
            sys.stderr.write(f"\r{stage}: {i}/{total or '?'}")
            sys.stderr.flush()

    res = run(cfg, progress)
    sys.stderr.write("\n")
    s, g = res.summary, res.summary["gait"]
    print(f"Results in: {res.output_dir}")
    print(f"  horse detected: {s['horse_detected_pct']:.0f}% of frames | rider: {s['rider_present_pct']:.0f}%")
    freq = g["stride_frequency_hz"]
    print(f"  gait estimate: {g['gait_estimate']} ({g['gait_confidence']})"
          + (f" | stride frequency {freq:.2f} Hz" if freq else ""))
    for name, r in g["joint_rom_deg"].items():
        if r:
            print(f"  {name:12s} range {r['min']:6.1f}-{r['max']:6.1f} deg (ROM {r['rom']:.1f})")
    if res.video_path:
        print(f"  annotated video: {res.video_path}")
    return 0
