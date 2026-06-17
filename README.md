# TemPose-BadmintonActionRecognition
Repository containing the implementation of TemPose: a skeleton-based transformer architecture for fine-grained action recognition for badminton strokes.

---

# Badminton Olympics Data Description

Covers how to create the **Badminton Olympics** dataset from publicly available videos and annotations.

---

## Overview

The pipeline turns full broadcast match videos into **stroke-level samples**.
Each stroke/action sample consists of:

- **Pose sequence**: 2 players, 17 joints (COCO convention), variable number of frames
- **Position sequence**: the same 2 players' ground-plane location (ankles projected via homography)
- **Shuttle trajectory**: shuttlecock `(x, y)` per frame, variable length
- **Label**: `[stroke_type, match_name]` per stroke

Sequence lengths vary per stroke, so all artifacts are saved as **`.pkl`** (lists of
variable-length arrays)

The loader expects four files:

```python
poses = pd.read_pickle('Data/2d_poses.pkl')
position = pd.read_pickle('Data/2d_positions.pkl')
shuttle = pd.read_pickle('Data/2d_shuttle-position.pkl')
labels = pd.read_pickle('Data/2d_labels.pkl')
```

---

## Data Source

- **Videos**: 10 full broadcast matches from the **2012 Olympics** (available on YouTube).
- **Annotations**: Created with the **ELAN** annotation tool and linked via the
  IIIT-H badminton analytics project:
  - https://cvit.iiit.ac.in/research/projects/cvit-projects/badminton-analytics
- After downloading the videos, the ELAN annotations are converted to a `.txt` file
  with a **start time** and **end time** for each action/stroke.

---

## Off-the-shelf inference tools

| Purpose            | Tool                                                                                 |
|--------------------|--------------------------------------------------------------------------------------|
| Pose estimation    | [MMPose](https://github.com/open-mmlab/mmpose/tree/main) — **HRNet**                  |
| Shuttle tracking   | [WASB-SBDT](https://github.com/nttcom/WASB-SBDT) (TrackNetV2 originally; WASB similar)|

---

## 1) Download and Prepare Match Videos

1. Collect the 10 full broadcast videos from YouTube.
2. Use a clear, stable naming convention per match (e.g. `match_01`, `match_02`, ...),
   because labels are per-stroke and must map back to their source video.
3. Record the original FPS so frame<->time mappings stay correct.

---

## 2) Export ELAN Annotations

1. Open each ELAN annotation file (`.eaf`).
2. Export to text/CSV containing at least:
   - `action_type`
   - `start_time` (or start frame)
   - `end_time` (or end frame)
3. Convert times to frame indices if needed:
   - `frame = round(time_seconds * fps)`

Olympics labels are **simple action strings** per stroke segment.

---

## 3) Extract Pose Data (MMPose + HRNet)

Run MMPose with **HRNet** (COCO 17-joint convention) on every frame of each match video.

For each frame:
1. Run pose estimation.
2. Keep detections for the two on-court players (see step 4 for filtering).
3. Store keypoints in image coordinates `(x, y)` per joint.

---

## 4) Compute Court Homography and Filter Players

For each match camera view:

1. Estimate the **homography** from image court points to the real court plane,
   using the known **court dimensions** and the corresponding points in the image.
2. **Transform the ankle joints to the ground plane** using the homography — this gives
   the **position data**.
3. Use the ground-plane ankle positions to **filter out non-player skeletons**: keep only
   detections whose feet/ankles fall **inside the court bounds**.

This removes audience / referee / false detections and stabilizes player identity.

The resulting ground-plane position data has the form:

```
[#strokes, #players (2), #frames_in_stroke, 2]   # (x, y) ground-plane coordinates
```

> Use `#players = 2` for singles. Use a larger value only if you extend this to doubles.

---

## 5) Slice Stroke-Level Pose Sequences

Using the exported stroke segments (`start_frame`, `end_frame`):

1. For each stroke/action, slice the pose frames.
2. Per-stroke shape:
   - `[#players, #frames_in_stroke, #joints, 2]` with `#players = 2`, `#joints = 17`.

Across all strokes, the pose dataset is:

```
[#strokes, #players (2), #frames_in_stroke, 17, 2]   # variable #frames per stroke
```

Because frame counts vary per stroke, save as a list-like object in **`.pkl`**.

---

## 6) Extract Shuttle Trajectories

Track the shuttle with WASB (or TrackNetV2). For each stroke segment, save the trajectory:

- Per stroke: `[#frames_in_stroke, 2]` for `(x, y)` image coordinates.

Across all strokes:

```
[#strokes, #frames_in_stroke, 2]   # variable length; prediction confidence NOT stored
```

---

## 7) Labels

Each stroke label is a **2-element entry** — the stroke type and the match name:

```python
labels[i] = [stroke_type, match_name]    # final shape: [#strokes, 2]

```

- **Element 0** (`stroke_type`): the action-type string used for classification
  (one of the 13 classes below).
- **Element 1** (`match_name`): identifies the source match. It is used to split
  train/test **by match**.

The raw ELAN tokens map to **13 action classes**. The suffix convention is
`...pt` = **player top** (player at top of court), `...pb` = **player bottom**
(player at bottom of court).

| Class | Token(s)                                   | Meaning (top / bottom player) |
|------:|--------------------------------------------|-------------------------------|
| 0     | `n`                                        | inaction (between rallies)    |
| 1     | `rtpt`                                     | react - top                   |
| 2     | `rtpb` / `rptb`                            | react - bottom                |
| 3     | `spt`                                      | serve - top                   |
| 4     | `spb`                                      | serve - bottom                |
| 5     | `smpt`                                     | smash - top                   |
| 6     | `smpb`                                     | smash - bottom                |
| 7     | `lbpt` / `lpbt`                            | lob - top                     |
| 8     | `lbpb`                                     | lob - bottom                  |
| 9     | `fhpt`                                     | forehand - top                |
| 10    | `fhpb`                                     | forehand - bottom             |
| 11    | `bhpt`                                     | backhand - top                |
| 12    | `bhpb`                                     | backhand - bottom             |

> **Note:** `rt` (**react**) is not a stroke. It marks the in-between-hits state and is
> meant to capture/simulate how much stress the player is under between shots.

The encoding tolerates trailing/leading whitespace and a few known typos in the raw
annotations:

```python
def one_hot_ol(labels_arr):
    y = []
    for case in labels_arr:
        if case[0] == ' n' or case[0] == 'n' or case[0] == 'n ':
            y.append(0)
        elif case[0] == 'rtpt' or case[0] == 'rtpt ':
            y.append(1)
        elif case[0] == 'rtpb' or case[0] == 'rptb' or case[0] == 'rtpb rtpb' or case[0] == 'rtpb ':
            y.append(2)
        elif case[0] == 'spt':
            y.append(3)
        elif case[0] == 'spb':
            y.append(4)
        elif case[0] == 'smpt':
            y.append(5)
        elif case[0] == 'smpb':
            y.append(6)
        elif case[0] == 'lbpt' or case[0] == 'lpbt':
            y.append(7)
        elif case[0] == 'lbpb' or case[0] == 'lbpb ':
            y.append(8)
        elif case[0] == 'fhpt':
            y.append(9)
        elif case[0] == 'fhpb':
            y.append(10)
        elif case[0] == 'bhpt' or case[0] == 'bhpt ':
            y.append(11)
        elif case[0] == 'bhpb':
            y.append(12)
        else:
            print(case[0])      # surfaces any unmapped / unexpected token
    return np.array(y)
```

---

## Final Artifacts


| File                              | Shape                                  | Contents |
|-----------------------------------|----------------------------------------|----------|
| `2d_poses.pkl`               | `[#strokes, 2, #frames, 17, 2]`        | Joint keypoints (image px) |
| `2d_positions.pkl`           | `[#strokes, 2, #frames, 2]`            | Ground-plane player positions |
| `2d_shuttle-position.pkl`    | `[#strokes, #frames, 2]`               | Shuttle trajectory (image px) |
| `2d_labels.pkl`              | `[#strokes, 2]`                        | `[stroke_type, match_name]` per stroke |

---

If you find the paper or repository helpful or interesting, please cite our work as:

```
@inproceedings{tempose,
  author={Ibh, Magnus and Grasshof, Stella and Witzner, Dan and Madeleine, Pascal},
  booktitle={2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW)},
  title={TemPose: a new skeleton-based transformer model designed for fine-grained motion recognition in badminton},
  year={2023},
  volume={},
  number={},
  pages={5199-5208},
  keywords={Visualization;Dynamics;Computer architecture;Transformers;Skeleton;Data models;Pattern recognition},
  doi={10.1109/CVPRW59228.2023.00548}}
```
