from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from ..layouts import KeypointLayout


class PoseBackend(ABC):
    """Estimates the keypoints of one horse given its bounding box."""

    layout: KeypointLayout

    @abstractmethod
    def estimate(self, frame: np.ndarray, box: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(xy, conf)``: (K, 2) full-frame pixel coords and (K,) scores."""

    def _empty(self) -> tuple[np.ndarray, np.ndarray]:
        k = self.layout.num_keypoints
        return np.full((k, 2), np.nan, np.float32), np.zeros(k, np.float32)
