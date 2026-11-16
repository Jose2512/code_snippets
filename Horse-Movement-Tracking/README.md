# Horse Movement Tracking

A computer vision app that takes a horse video, **with or without a rider**, and tracks:

- **Leg joints and hooves** for all four legs: joint angles, how far each limb swings forward and back, hoof trajectories
- **Head position**: nose height, neck elevation, face angle from the vertical, head bob
- **Gait**: stride frequency, phase between limbs, duty factor, a walk/trot/canter/pace estimate, left/right asymmetry

You get an annotated video, CSV files for every frame and keypoint, a JSON summary and charts.

```
video ─► YOLO11-seg (horse + person) ─► ByteTrack (follow one horse)
                                           │
                    rider association ◄────┤  (person on the horse's back vs handler on foot)
                                           ▼
                       top-down pose model on the horse box (AP-10K, 17 keypoints)
                                           ▼
     reject limb points hidden by the rider ─► fill short gaps ─► Savitzky–Golay smoothing
                                           ▼
          kinematics per frame ─► gait analysis per clip ─► video / CSV / JSON / plots
```

## Install

```bash
cd Horse-Movement-Tracking
pip install -r requirements.txt          # or: pip install -e .[ui]
python scripts/download_models.py        # YOLO11n-seg + ViTPose-B AP-10K (ONNX) into ./models
```

`download_models.py --size s|b|l|h` chooses the ViTPose size. Smaller is faster, larger is more accurate.
If the download fails, get the `.onnx` file from
<https://huggingface.co/JunkyByte/easy_ViTPose/tree/main/onnx/ap10k> and save it in `models/`.

## Use

**Command line**

```bash
python -m horsetrack clip.mp4 -o outputs/clip \
    --detector models/yolo11n-seg.pt --pose-model models/vitpose-b-ap10k.onnx
```

Useful options:

| option | purpose |
|---|---|
| `--device cuda` | run on GPU |
| `--horse-id N` | follow a specific horse when several are in view (IDs appear in `metrics.csv`) |
| `--every 2` | analyse every 2nd frame (faster) |
| `--mask-rider` | paint the rider out before pose estimation (can help when the rider's legs confuse the model) |
| `--kpt-thr 0.3` | minimum keypoint confidence |
| `--layout configs/horse_detailed_layout.json` | use a custom keypoint layout (needs a matching model) |
| `--no-video` | data only, no annotated video |

**Web app**

```bash
streamlit run app.py
```

Upload a clip, click **Analyse**, then view the video, metrics and charts and download the results.

**Python**

```python
from horsetrack import Config, run
res = run(Config(video="clip.mp4", pose_model="models/vitpose-b-ap10k.onnx", detector="models/yolo11n-seg.pt"))
print(res.summary["gait"]["gait_estimate"], res.metrics.head())
```

## Outputs

| file | content |
|---|---|
| `annotated.mp4` | skeleton colour-coded by limb, joint angles, head vector, hoof trails, horse/rider boxes, info panel |
| `keypoints.csv` | per frame and keypoint: raw x/y, confidence, smoothed x/y, status (`detected` / `interpolated` / `occluded_by_rider` / `missing`) |
| `metrics.csv` | per frame: rider present, joint angles, limb protraction, body length, nose position, head height, neck elevation, face inclination |
| `summary.json` | stride frequency, gait estimate, inter-limb phase lags, duty factor, hoof strikes, joint ROM, left/right asymmetry, head summary, keypoint visibility |
| `plots/*.png` | joint angles, limb protraction, head position, detection timeline |

### Measures

All values are **2-D projections**. Film from the side, with the camera roughly perpendicular to the direction of travel.
Measures are taken relative to the horse's facing direction, so the same numbers come out whether it moves left or right.

- **Joint angle**: interior angle at a joint, where 180° means fully extended.
- **Protraction**: angle of the shoulder/hip→hoof line from the vertical. Positive means the hoof is ahead of the shoulder/hip, negative means behind. Panning the camera doesn't change it, so it drives the gait analysis: the limb rotates backward during stance and forward during swing.
- **Head height**: height of the nose above the neck point, in body lengths (BL).
- **Neck elevation**: angle of the neck→nose line above the body axis.
- **Face inclination**: angle of the eyes/poll→nose line from the vertical. 0° means the face is vertical, positive means in front of the vertical, negative means behind it.
- **Head bob**: 5th–95th percentile range of head height across the clip.

## With or without a rider

- Every person detected is checked against the horse. A **rider** overlaps the upper half of the horse box with their feet around the belly line. A **handler** on foot has their feet at ground level and is ignored.
- The rider flag is smoothed over about 0.5 s so it doesn't flicker.
- When a rider is present, any limb keypoint on the rider's segmentation mask must reach a higher confidence (`--rider-kpt-thr`). Below that, it counts as the rider's leg or a guess behind it, and is dropped and interpolated.
- Optionally, `--mask-rider` removes the rider from the image before pose estimation.

## Pose models and keypoint layouts

| backend | model | keypoints |
|---|---|---|
| `vitpose` (default) | ONNX top-down heatmap model, e.g. ViTPose AP-10K | AP-10K: eyes, nose, neck, tail root, shoulder/elbow/hoof per front leg, hip/knee/hoof per hind leg |
| `yolo` | your own YOLO-pose model (`scripts/train_yolo_pose.py`) | any layout, e.g. the 27-point equine layout |
| `mmpose` | RTMPose / ViTPose AP-10K checkpoints (needs `mmpose`, `mmcv`, `mmdet`) | AP-10K |

AP-10K gives one middle joint per leg. For full equine joint detail (**carpus, fetlock, stifle, hock, poll, withers**), annotate your own frames with the layout in `configs/horse_detailed_layout.json` and train a YOLO-pose model:

```bash
python scripts/train_yolo_pose.py --data configs/horse_pose_dataset.yaml   # edit kpt_shape/flip_idx for 27 points
python -m horsetrack clip.mp4 --pose-backend yolo --pose-model runs/pose/train/weights/best.pt \
       --layout configs/horse_detailed_layout.json
```

Layouts are plain JSON (keypoints, skeleton, joint angles, limb chains, head and body axis), so the analysis and drawing adapt automatically.

## Tests

```bash
pip install pytest torch onnx
python -m pytest -q tests
```

The tests cover geometry, smoothing, gait classification on synthetic walk/trot/pace/canter signals, rider vs handler association, track following, and the ONNX backend's mapping from heatmaps back to pixels.

## Limitations

- Accuracy depends on the pose model. Public AP-10K models cover many animal species and are not specialised for horses. Fine-tune on your own footage for clinical-grade work.
- Single camera, 2-D only. Angles are distorted when the horse moves toward or away from the camera.
- The gait label and asymmetry figures are **estimates for orientation, not a veterinary diagnosis**.
