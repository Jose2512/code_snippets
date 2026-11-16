"""Checks the ONNX heatmap backend maps heatmap peaks back to the right pixels,
using a tiny fake model that always outputs known peaks."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("onnx")

from horsetrack.layouts import AP10K  # noqa: E402
from horsetrack.pose.vitpose_onnx import HeatmapOnnxPose  # noqa: E402

HM_W, HM_H = 48, 64
PEAKS = [(4 + 2 * k, 6 + 3 * k) for k in range(17)]  # (hx, hy) per keypoint


class FakeHeatmapModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        yy, xx = np.mgrid[0:HM_H, 0:HM_W]
        hms = np.stack([np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / 2.0) for x, y in PEAKS]).astype(np.float32)
        self.register_buffer("hms", torch.from_numpy(hms)[None])

    def forward(self, x):
        return self.hms + 0.0 * x.mean()


@pytest.fixture(scope="module")
def onnx_path(tmp_path_factory):
    p = tmp_path_factory.mktemp("m") / "fake.onnx"
    torch.onnx.export(FakeHeatmapModel(), torch.zeros(1, 3, 256, 192), str(p),
                      input_names=["input"], output_names=["heatmaps"], dynamo=False)
    return str(p)


def test_peaks_map_back_to_frame(onnx_path):
    pose = HeatmapOnnxPose(onnx_path, AP10K, padding=1.0)
    assert (pose.in_w, pose.in_h) == (192, 256)
    frame = np.zeros((480, 640, 3), np.uint8)
    # box with exactly the model aspect ratio and 2x the model size
    box = np.array([100, 50, 100 + 384, 50 + 512], np.float32)
    xy, conf = pose.estimate(frame, box)
    expected = np.array([(100 + hx * 4 * 2, 50 + hy * 4 * 2) for hx, hy in PEAKS], np.float32)
    np.testing.assert_allclose(xy, expected, atol=1e-3)
    assert (conf > 0.99).all()


def test_layout_mismatch_is_reported(onnx_path):
    from horsetrack.layouts import load_layout

    pose = HeatmapOnnxPose(onnx_path, load_layout("configs/horse_detailed_layout.json"))
    with pytest.raises(ValueError, match="keypoints"):
        pose.estimate(np.zeros((100, 100, 3), np.uint8), np.array([0, 0, 50, 50]))
