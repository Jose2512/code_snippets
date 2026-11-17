"""Keypoint layouts (which joints a pose model predicts and how they relate).

A layout is data-driven so the rest of the app (kinematics, gait analysis,
drawing) works with any pose model: the built-in AP-10K layout used by the
pretrained animal models, or a custom, more detailed equine layout loaded
from JSON (see ``configs/horse_detailed_layout.json``).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

LIMB_KEYS = ("LF", "RF", "LH", "RH")  # left/right fore, left/right hind


@dataclass(frozen=True)
class JointAngle:
    """Interior angle (degrees) measured at ``vertex`` between ``a`` and ``c``."""

    name: str
    a: str
    vertex: str
    c: str


@dataclass(frozen=True)
class KeypointLayout:
    name: str
    keypoints: tuple[str, ...]
    skeleton: tuple[tuple[str, str], ...]
    angles: tuple[JointAngle, ...]
    # Limb chains ordered proximal -> distal; the last point is the hoof.
    limbs: dict[str, tuple[str, ...]]
    # Head definition: nose (required), neck/withers reference (required),
    # optional poll and eyes for face inclination.
    nose: str
    neck: str
    poll: str | None = None
    eyes: tuple[str, ...] = ()
    # Body axis (front, rear), e.g. ("Neck", "Root_of_tail").
    body_axis: tuple[str, str] = ("Neck", "Root_of_tail")
    # Points used for vertical-movement asymmetry (default: ends of the body axis).
    withers: str | None = None
    pelvis: str | None = None
    flip_pairs: tuple[tuple[str, str], ...] = field(default=())

    def __post_init__(self) -> None:
        names = set(self.keypoints)
        refs = [n for pair in self.skeleton for n in pair]
        refs += [n for a in self.angles for n in (a.a, a.vertex, a.c)]
        refs += [n for chain in self.limbs.values() for n in chain]
        refs += [self.nose, self.neck, *self.eyes, *self.body_axis]
        refs += [n for n in (self.withers, self.pelvis) if n]
        if self.poll:
            refs.append(self.poll)
        unknown = sorted(set(refs) - names)
        if unknown:
            raise ValueError(f"Layout '{self.name}' references unknown keypoints: {unknown}")
        missing_limbs = [k for k in LIMB_KEYS if k not in self.limbs]
        if missing_limbs:
            raise ValueError(f"Layout '{self.name}' must define limbs {missing_limbs}")

    @property
    def num_keypoints(self) -> int:
        return len(self.keypoints)

    def idx(self, name: str) -> int:
        return self.keypoints.index(name)

    @property
    def withers_point(self) -> str:
        return self.withers or self.body_axis[0]

    @property
    def pelvis_point(self) -> str:
        return self.pelvis or self.body_axis[1]

    @property
    def hooves(self) -> dict[str, str]:
        return {k: chain[-1] for k, chain in self.limbs.items()}

    def part_of(self, name: str) -> str:
        """Which body part a keypoint belongs to (limb key, or 'axial')."""
        for key, chain in self.limbs.items():
            if name in chain:
                return key
        return "axial"

    # ------------------------------------------------------------------ I/O
    @classmethod
    def from_dict(cls, d: dict) -> "KeypointLayout":
        head = d.get("head", {})
        return cls(
            name=d["name"],
            keypoints=tuple(d["keypoints"]),
            skeleton=tuple(tuple(p) for p in d["skeleton"]),
            angles=tuple(JointAngle(a["name"], *a["points"]) for a in d.get("angles", [])),
            limbs={k: tuple(v) for k, v in d["limbs"].items()},
            nose=head["nose"],
            neck=head["neck"],
            poll=head.get("poll"),
            eyes=tuple(head.get("eyes", ())),
            body_axis=tuple(d["body_axis"]),
            withers=d.get("withers"),
            pelvis=d.get("pelvis"),
            flip_pairs=tuple(tuple(p) for p in d.get("flip_pairs", [])),
        )

    @classmethod
    def from_json(cls, path: str | Path) -> "KeypointLayout":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


# AP-10K (17 keypoints) — the layout of the public pretrained animal pose
# models (ViTPose / RTMPose AP-10K). On the front leg AP-10K calls the middle
# joint "Elbow"; on the hind leg "Knee" (anatomically closest to the
# stifle/hock region depending on annotation). Paws are the hooves.
AP10K = KeypointLayout.from_dict(
    {
        "name": "ap10k",
        "keypoints": [
            "L_Eye", "R_Eye", "Nose", "Neck", "Root_of_tail",
            "L_Shoulder", "L_Elbow", "L_F_Paw",
            "R_Shoulder", "R_Elbow", "R_F_Paw",
            "L_Hip", "L_Knee", "L_B_Paw",
            "R_Hip", "R_Knee", "R_B_Paw",
        ],
        "skeleton": [
            ["L_Eye", "R_Eye"], ["L_Eye", "Nose"], ["R_Eye", "Nose"],
            ["Nose", "Neck"], ["Neck", "Root_of_tail"],
            ["Neck", "L_Shoulder"], ["L_Shoulder", "L_Elbow"], ["L_Elbow", "L_F_Paw"],
            ["Neck", "R_Shoulder"], ["R_Shoulder", "R_Elbow"], ["R_Elbow", "R_F_Paw"],
            ["Root_of_tail", "L_Hip"], ["L_Hip", "L_Knee"], ["L_Knee", "L_B_Paw"],
            ["Root_of_tail", "R_Hip"], ["R_Hip", "R_Knee"], ["R_Knee", "R_B_Paw"],
        ],
        "angles": [
            {"name": "LF_elbow", "points": ["L_Shoulder", "L_Elbow", "L_F_Paw"]},
            {"name": "RF_elbow", "points": ["R_Shoulder", "R_Elbow", "R_F_Paw"]},
            {"name": "LH_knee", "points": ["L_Hip", "L_Knee", "L_B_Paw"]},
            {"name": "RH_knee", "points": ["R_Hip", "R_Knee", "R_B_Paw"]},
        ],
        "limbs": {
            "LF": ["L_Shoulder", "L_Elbow", "L_F_Paw"],
            "RF": ["R_Shoulder", "R_Elbow", "R_F_Paw"],
            "LH": ["L_Hip", "L_Knee", "L_B_Paw"],
            "RH": ["R_Hip", "R_Knee", "R_B_Paw"],
        },
        "head": {"nose": "Nose", "neck": "Neck", "eyes": ["L_Eye", "R_Eye"]},
        "body_axis": ["Neck", "Root_of_tail"],
        "flip_pairs": [
            ["L_Eye", "R_Eye"], ["L_Shoulder", "R_Shoulder"], ["L_Elbow", "R_Elbow"],
            ["L_F_Paw", "R_F_Paw"], ["L_Hip", "R_Hip"], ["L_Knee", "R_Knee"],
            ["L_B_Paw", "R_B_Paw"],
        ],
    }
)

BUILTIN_LAYOUTS = {"ap10k": AP10K}


def load_layout(spec: str | Path | None) -> KeypointLayout:
    """Resolve a layout from a built-in name or a JSON file path."""
    if spec is None:
        return AP10K
    if str(spec).lower() in BUILTIN_LAYOUTS:
        return BUILTIN_LAYOUTS[str(spec).lower()]
    return KeypointLayout.from_json(spec)
