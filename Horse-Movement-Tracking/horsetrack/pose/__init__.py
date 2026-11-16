from __future__ import annotations

from ..layouts import KeypointLayout
from .base import PoseBackend

BACKENDS = ("vitpose", "yolo", "mmpose")


def create_pose_backend(name: str, layout: KeypointLayout, model_path: str | None = None,
                        device: str | None = None, mmpose_config: str | None = None) -> PoseBackend:
    name = name.lower()
    if name in ("vitpose", "onnx"):
        if not model_path:
            raise ValueError("The vitpose backend needs --pose-model path/to/model.onnx "
                             "(run scripts/download_models.py).")
        from .vitpose_onnx import HeatmapOnnxPose

        return HeatmapOnnxPose(model_path, layout, device=device)
    if name == "yolo":
        if not model_path:
            raise ValueError("The yolo backend needs --pose-model path/to/horse-pose.pt")
        from .yolo_pose import YoloPose

        return YoloPose(model_path, layout, device=device)
    if name == "mmpose":
        from .mmpose_backend import MMPosePose

        return MMPosePose(layout, config=mmpose_config, checkpoint=model_path, device=device)
    raise ValueError(f"Unknown pose backend '{name}'. Choose from {BACKENDS}.")


__all__ = ["PoseBackend", "create_pose_backend", "BACKENDS"]
