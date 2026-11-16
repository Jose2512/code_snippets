"""Top-down heatmap pose model in ONNX format (e.g. ViTPose AP-10K).

Works with any ONNX model that takes a normalised RGB crop ``[1, 3, H, W]``
and returns heatmaps ``[1, K, h, w]`` — the format of the ViTPose
animal models (AP-10K / APT-36K) distributed by the easy_ViTPose project,
and of most MMPose heatmap models exported with mmdeploy.
"""
from __future__ import annotations

import numpy as np
import cv2

from ..layouts import KeypointLayout
from .base import PoseBackend

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)


def box_to_center_size(box: np.ndarray, aspect: float, padding: float) -> tuple[np.ndarray, np.ndarray]:
    """Expand a box to the model aspect ratio (w / h) and add padding."""
    x1, y1, x2, y2 = [float(v) for v in box]
    center = np.array([(x1 + x2) / 2, (y1 + y2) / 2], np.float32)
    w, h = max(x2 - x1, 1.0), max(y2 - y1, 1.0)
    if w > aspect * h:
        h = w / aspect
    else:
        w = h * aspect
    return center, np.array([w * padding, h * padding], np.float32)


def heatmaps_to_coords(heatmaps: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Argmax + quarter-pixel refinement. heatmaps (K, h, w) -> (K, 2) xy, (K,) score."""
    k, h, w = heatmaps.shape
    flat = heatmaps.reshape(k, -1)
    idx = flat.argmax(axis=1)
    score = flat[np.arange(k), idx]
    xs = (idx % w).astype(np.float32)
    ys = (idx // w).astype(np.float32)
    for j in range(k):
        x, y = int(xs[j]), int(ys[j])
        hm = heatmaps[j]
        if 0 < x < w - 1:
            xs[j] += 0.25 * np.sign(hm[y, x + 1] - hm[y, x - 1])
        if 0 < y < h - 1:
            ys[j] += 0.25 * np.sign(hm[y + 1, x] - hm[y - 1, x])
    return np.stack([xs, ys], axis=1), score.astype(np.float32)


class HeatmapOnnxPose(PoseBackend):
    def __init__(
        self,
        model_path: str,
        layout: KeypointLayout,
        device: str | None = None,
        padding: float = 1.25,
        input_size: tuple[int, int] | None = None,  # (w, h) when the model has dynamic dims
        mean: np.ndarray = IMAGENET_MEAN,
        std: np.ndarray = IMAGENET_STD,
        use_udp: bool = False,
    ) -> None:
        import onnxruntime as ort

        providers = ["CPUExecutionProvider"]
        if device and device != "cpu" and "CUDAExecutionProvider" in ort.get_available_providers():
            providers.insert(0, "CUDAExecutionProvider")
        self.session = ort.InferenceSession(model_path, providers=providers)
        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        shape = inp.shape  # [1, 3, H, W], possibly symbolic
        h = shape[2] if isinstance(shape[2], int) else None
        w = shape[3] if isinstance(shape[3], int) else None
        if input_size:
            w, h = input_size
        self.in_w, self.in_h = (w or 192), (h or 256)
        self.layout = layout
        self.padding = padding
        self.mean, self.std = mean, std
        self.use_udp = use_udp

    def _preprocess(self, frame: np.ndarray, box: np.ndarray):
        center, size = box_to_center_size(box, self.in_w / self.in_h, self.padding)
        s = self.in_w / size[0]
        m = np.array([[s, 0, self.in_w / 2 - s * center[0]],
                      [0, s, self.in_h / 2 - s * center[1]]], np.float32)
        crop = cv2.warpAffine(frame, m, (self.in_w, self.in_h), flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
        x = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        x = (x - self.mean) / self.std
        return x.transpose(2, 0, 1)[None], m

    def estimate(self, frame: np.ndarray, box: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x, m = self._preprocess(frame, box)
        heatmaps = self.session.run(None, {self.input_name: x})[0][0]
        if heatmaps.shape[0] != self.layout.num_keypoints:
            raise ValueError(
                f"Pose model outputs {heatmaps.shape[0]} keypoints but layout "
                f"'{self.layout.name}' has {self.layout.num_keypoints}. Use --layout to match the model."
            )
        hm_h, hm_w = heatmaps.shape[1:]
        coords, score = heatmaps_to_coords(heatmaps)
        if self.use_udp:
            coords[:, 0] *= (self.in_w - 1) / (hm_w - 1)
            coords[:, 1] *= (self.in_h - 1) / (hm_h - 1)
        else:
            coords[:, 0] *= self.in_w / hm_w
            coords[:, 1] *= self.in_h / hm_h
        s, tx, ty = m[0, 0], m[0, 2], m[1, 2]
        xy = np.stack([(coords[:, 0] - tx) / s, (coords[:, 1] - ty) / s], axis=1)
        return xy.astype(np.float32), np.clip(score, 0.0, 1.0)
