"""Charts of the analysed clip (PNG)."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

LIMB_COLORS = {"LF": "#f2a900", "RF": "#2a7de1", "LH": "#3aa655", "RH": "#c040c0"}


def _limb_of(col: str) -> str | None:
    parts = col.split("_")
    return parts[1] if len(parts) > 1 and parts[1] in LIMB_COLORS else None


def make_plots(metrics: pd.DataFrame, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    t = metrics["time_s"]
    paths = []

    angle_cols = [c for c in metrics.columns if c.startswith("angle_")]
    if angle_cols:
        fig, ax = plt.subplots(figsize=(11, 4.5))
        seen: dict[str | None, int] = {}
        for c in angle_cols:
            limb = _limb_of(c)
            ls = ("-", "--", ":", "-.")[seen.get(limb, 0) % 4]
            seen[limb] = seen.get(limb, 0) + 1
            ax.plot(t, metrics[c], ls, lw=1.4, label=c[6:], color=LIMB_COLORS.get(limb))
        ax.set(xlabel="time (s)", ylabel="joint angle (°)", title="Joint angles")
        ax.legend(ncol=4, fontsize=8)
        ax.grid(alpha=0.3)
        paths.append(_save(fig, out_dir / "joint_angles.png"))

    prot = [c for c in metrics.columns if c.startswith("protraction_")]
    if prot:
        fig, ax = plt.subplots(figsize=(11, 4.5))
        for c in prot:
            k = c.split("_")[1]
            ax.plot(t, metrics[c], lw=1.4, label=k, color=LIMB_COLORS[k])
        ax.axhline(0, color="k", lw=0.6)
        ax.set(xlabel="time (s)", ylabel="limb angle from vertical (°)\n+ protraction / − retraction",
               title="Limb protraction–retraction")
        ax.legend()
        ax.grid(alpha=0.3)
        paths.append(_save(fig, out_dir / "limb_protraction.png"))

    head_cols = [c for c in ("head_height", "neck_elevation", "face_inclination") if c in metrics]
    if head_cols:
        fig, axes = plt.subplots(len(head_cols), 1, figsize=(11, 2.4 * len(head_cols)), sharex=True)
        axes = [axes] if len(head_cols) == 1 else axes
        labels = {"head_height": "nose height\n(body lengths)", "neck_elevation": "neck elevation (°)",
                  "face_inclination": "face vs vertical (°)"}
        for ax, c in zip(axes, head_cols):
            ax.plot(t, metrics[c], color="#1b6f8a", lw=1.4)
            ax.set_ylabel(labels[c])
            ax.grid(alpha=0.3)
        axes[0].set_title("Head position")
        axes[-1].set_xlabel("time (s)")
        paths.append(_save(fig, out_dir / "head_position.png"))

    if "rider_present" in metrics:
        fig, ax = plt.subplots(figsize=(11, 1.6))
        ax.fill_between(t, metrics["horse_detected"].astype(float), step="mid", alpha=0.3, label="horse detected")
        ax.fill_between(t, metrics["rider_present"].astype(float) * 0.6, step="mid", alpha=0.6, label="rider", color="#ff8c28")
        ax.set(yticks=[], xlabel="time (s)", title="Detection timeline")
        ax.legend(loc="upper right", fontsize=8)
        paths.append(_save(fig, out_dir / "timeline.png"))
    return paths


def _save(fig, path: Path) -> Path:
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path
