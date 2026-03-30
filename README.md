# Real-Time Posture Detection System

A webcam-based computer vision application that monitors sitting posture
in real time and alerts the user when bad posture is detected.
Built using MediaPipe BlazePose and OpenCV.

![Python](https://img.shields.io/badge/Python-3.8+-blue)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10.x-orange)

---

## The Problem

Students and desk workers sit for 6–8 hours daily, often developing
poor posture habits without realizing it. Every 2.5 cm of forward head
displacement adds ~4.5 kg of effective load on the cervical spine.
No affordable real-time solution exists for personal posture monitoring.

---

## What It Does

- Detects 33 body landmarks per frame using MediaPipe BlazePose
- Calculates neck inclination angle (nose → ear → shoulder)
- Measures shoulder symmetry (vertical pixel difference)
- Flags bad posture when BOTH metrics exceed calibrated thresholds
- Shows green / red status bar live on webcam feed
- Fires alert after 30 consecutive bad frames (~1 second)
- Runs entirely on CPU — no GPU required

---

## Tech Stack

| Tool | Version | Purpose |
|---|---|---|
| Python | 3.8+ | Core language |
| MediaPipe | 0.10.x | Pose landmark detection |
| OpenCV | 4.x | Webcam, visualization |
| NumPy | 1.x | Angle calculation |

---

## Project Structure
```
posture-detection/
│
├── posture_detect.py       ← Main application
├── angles.py               ← Angle calculation logic
├── debug_angles.py         ← Diagnostic tool
├── test_setup.py           ← Installation checker
├── requirements.txt        ← Dependencies
├── pose_landmarker.task    ← MediaPipe model (auto-downloaded)
└── screenshots/
    ├── good_posture.png
    └── bad_posture.png
```

---

## Setup

**Step 1 — Clone the repository**
```bash
git clone https://github.com/YOUR_USERNAME/posture-detection.git
cd posture-detection
```

**Step 2 — Install dependencies**
```bash
pip install -r requirements.txt
```

**Step 3 — Verify installation**
```bash
python test_setup.py
```

You should see:
```
OpenCV: 4.x.x
MediaPipe: 0.10.x
NumPy: x.x.x
Webcam: OK
```

---

## Usage
```bash
python posture_detect.py
```

The pose model (~3 MB) downloads automatically on first run.

**Controls:**
- `Q` — quit the application

**On screen you will see:**
- Green bar at top → good posture
- Red bar at top → bad posture alert
- Neck angle and shoulder tilt values (live)
- Alert threshold progress bar at bottom

---

## How It Works

### Landmark Selection

MediaPipe detects 33 body landmarks. This system uses:

| Landmark | Index | Role |
|---|---|---|
| Nose | 0 | Head position reference |
| Left ear | 7 | Neck angle vertex |
| Left shoulder | 11 | Angle + tilt reference |
| Right shoulder | 12 | Tilt measurement |

### Neck Angle Formula
```
angle = arccos( (BA · BC) / (|BA| × |BC|) )

A = nose
B = left ear  ← vertex
C = left shoulder
```

Good posture: 93–104°
Bad posture: below 93°

### Posture Decision
```python
is_bad = (neck_angle < 93) AND (shoulder_tilt > 16)
```

AND logic prevents false alerts from individual noisy readings.
A 30-frame buffer prevents flickering alerts from brief movements.

### Threshold Calibration

Default thresholds were calibrated empirically using the
`debug_angles.py` diagnostic tool. If you experience false alerts,
adjust these values in `posture_detect.py`:
```python
NECK_THRESHOLD    = 93    # degrees — lower = more lenient
TILT_THRESHOLD    = 16    # pixels  — higher = more lenient
ALERT_FRAME_LIMIT = 30    # frames  — higher = slower to alert
```

---

## Results

| Test | Condition | Result |
|---|---|---|
| 1 | Good posture, normal lighting | Correct |
| 2 | Deliberate slouch forward | Correct |
| 3 | Leaning to one side | Correct |
| 4 | Low light condition | Correct |
| 5 | Sitting far from camera (>1.5m) | Partial |

**9/10 correct detections. ~28 fps on CPU.**

---

## Limitations

- Works best with a single person in frame
- Camera should be at eye level or slightly below
- Performance drops in very low light conditions
- Cannot detect forward/back lean along the Z axis
  (depth camera like Intel RealSense required for this)

---

## Future Work

- Depth camera support for Z-axis head displacement detection
- Exercise posture monitoring (squat, plank, bicep curl)
- Audio alert alongside visual alert
- CSV logging for daily posture trend analysis
- Web interface for remote monitoring

---

## Course Details

| Field | Details |
|---|---|
| University | VIT Bhopal University |
| Course | Computer Vision (CSE3010) |
| Semester | 8 |
| Student | Puneet |
| Roll No | 23BAI10122 |
| Project Type | BYOP — Bring Your Own Project |

---

## References

- [MediaPipe Pose Landmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker)
- [OpenCV Documentation](https://docs.opencv.org)
- Bazarevsky et al. — BlazePose: On-device Real-time Body Pose Tracking. Google Research, 2020.
- Hansraj, K.K. (2014). Assessment of Stresses in the Cervical Spine. Surgical Technology International.
