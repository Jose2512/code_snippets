"""MMPose backend (RTMPose / ViTPose AP-10K checkpoints).

Optional: requires ``mmpose`` (and mmcv/mmdet). Without a config, the
MMPose ``animal`` alias (RTMPose-m trained on AP-10K) is downloaded.
"""
from __future__ import annotations

import numpy as np

from ..layouts import KeypointLayout
from .base import PoseBackend


class MMPosePose(PoseBackend):
    def __init__(self, layout: KeypointLayout, config: str | None = None,
                 checkpoint: str | None = None, device: str | None = None) -> None:
        try:
            from mmpose.apis import inference_topdown, init_model
        except ImportError as e:  # pragma: no cover - optional dependency
            raise ImportError("The mmpose backend needs `pip install mmpose mmcv mmdet` (see README).") from e
        self._infer = inference_topdown
        device = device or "cpu"
        if config:
            self.model = init_model(config, checkpoint, device=device)
        else:
            from mmpose.apis import MMPoseInferencer

            self.model = MMPoseInferencer(pose2d="animal", device=device).inferencer.model
        self.layout = layout

    def estimate(self, frame: np.ndarray, box: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        res = self._infer(self.model, frame, np.asarray([box], np.float32), bbox_format="xyxy")
        if not res:
            return self._empty()
        inst = res[0].pred_instances
        return (np.asarray(inst.keypoints[0], np.float32),
                np.clip(np.asarray(inst.keypoint_scores[0], np.float32), 0, 1))
