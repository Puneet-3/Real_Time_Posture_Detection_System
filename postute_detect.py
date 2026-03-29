# ─────────────────────────────────────────────────────────────
# Project   : Real-Time Posture Detection System
# Author    : Puneet
# Roll No   : 23BAI10122
# Course    : Computer Vision (BECE407L)
# Institute : VIT Bhopal University
# Semester  : 8
# ─────────────────────────────────────────────────────────────



import cv2
import mediapipe as mp
import numpy as np
import os
import urllib.request

from angles import (
    calculate_neck_angle,
    calculate_shoulder_tilt,
    is_bad_posture
)
MODEL_PATH = "pose_landmarker.task"
MODEL_URL  = (
    "https://storage.googleapis.com/mediapipe-models/"
    "pose_landmarker/pose_landmarker_lite/float16/latest/"
    "pose_landmarker_lite.task"
)

if not os.path.exists(MODEL_PATH):
    print("Downloading pose model (first run only)...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Model downloaded successfully.")

BaseOptions           = mp.tasks.BaseOptions
PoseLandmarker        = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode     = mp.tasks.vision.RunningMode

options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.IMAGE
)

NECK_THRESHOLD    = 93    
TILT_THRESHOLD    = 16    
ALERT_FRAME_LIMIT = 30    

bad_frame_count = 0
alert_active    = False

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    print("Try changing VideoCapture(0) to VideoCapture(1)")
    exit()

print("Posture Detection running... Press Q to quit.")

CONNECTIONS = [
    (0, 7),  (0, 8),
    (7, 11), (8, 12),
    (11, 12),
    (11, 13), (13, 15),
    (12, 14), (14, 16),
    (11, 23), (12, 24),
    (23, 24),
    (23, 25), (24, 26),
    (25, 27), (26, 28)
]

with PoseLandmarker.create_from_options(options) as landmarker:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Webcam frame not received. Exiting.")
            break

        h, w = frame.shape[:2]
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image  = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )
        result = landmarker.detect(mp_image)

        if result.pose_landmarks and len(result.pose_landmarks) > 0:
            lm = result.pose_landmarks[0]
            for a, b in CONNECTIONS:
                if a < len(lm) and b < len(lm):
                    x1 = int(lm[a].x * w)
                    y1 = int(lm[a].y * h)
                    x2 = int(lm[b].x * w)
                    y2 = int(lm[b].y * h)
                    cv2.line(frame, (x1, y1), (x2, y2),
                             (180, 180, 180), 2)

            for point in lm:
                cx = int(point.x * w)
                cy = int(point.y * h)
                cv2.circle(frame, (cx, cy), 4, (80, 80, 255), -1)

            for idx in [0, 7, 8, 11, 12, 23, 24]:
                if idx < len(lm):
                    cx = int(lm[idx].x * w)
                    cy = int(lm[idx].y * h)
                    cv2.circle(frame, (cx, cy), 7,
                               (0, 255, 255), -1)

            neck_angle    = calculate_neck_angle(lm, w, h)
            shoulder_tilt = calculate_shoulder_tilt(lm, w, h)
            bad, reasons  = is_bad_posture(
                neck_angle,
                shoulder_tilt,
                neck_threshold=NECK_THRESHOLD,
                tilt_threshold=TILT_THRESHOLD
            )

            if bad:
                bad_frame_count += 1
            else:
                bad_frame_count = max(0, bad_frame_count - 2)

            alert_active = bad_frame_count >= ALERT_FRAME_LIMIT

            bar_color   = (0, 0, 200) if alert_active else (0, 160, 0)
            status_text = "BAD POSTURE  Sit straight!" \
                          if alert_active else "Good posture"

            cv2.rectangle(frame, (0, 0), (w, 65), bar_color, -1)
            cv2.putText(frame, status_text,
                        (20, 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                        (255, 255, 255), 2, cv2.LINE_AA)

            cv2.rectangle(frame, (0, 70), (360, 200),
                          (30, 30, 30), -1)

            neck_color = (0, 255, 0) if neck_angle >= NECK_THRESHOLD \
                         else (0, 0, 255)
            cv2.putText(frame,
                        f"Neck angle : {neck_angle} deg",
                        (12, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                        neck_color, 2, cv2.LINE_AA)
            
            tilt_color = (0, 255, 0) if shoulder_tilt <= TILT_THRESHOLD \
                         else (0, 0, 255)
            cv2.putText(frame,
                        f"Shoulder tilt: {shoulder_tilt} px",
                        (12, 130),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                        tilt_color, 2, cv2.LINE_AA)
            cv2.putText(frame,
                        f"Neck thresh : {NECK_THRESHOLD} deg",
                        (12, 160),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (160, 160, 160), 1, cv2.LINE_AA)
            cv2.putText(frame,
                        f"Tilt thresh : {TILT_THRESHOLD} px",
                        (12, 182),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (160, 160, 160), 1, cv2.LINE_AA)
            bar_fill = int(
                (bad_frame_count / ALERT_FRAME_LIMIT) * 300
            )
            bar_fill      = min(bar_fill, 300)
            bar_fill_color = (0, 0, 220) if alert_active \
                             else (0, 180, 80)

            cv2.rectangle(frame,
                          (0, h - 35), (300, h - 10),
                          (50, 50, 50), -1)
            if bar_fill > 0:
                cv2.rectangle(frame,
                              (0, h - 35), (bar_fill, h - 10),
                              bar_fill_color, -1)
            cv2.putText(frame, "Alert threshold",
                        (8, h - 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (200, 200, 200), 1, cv2.LINE_AA)
            if alert_active and reasons:
                cv2.putText(frame,
                            reasons[0][:55],
                            (20, h - 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (100, 100, 255), 2, cv2.LINE_AA)

        else:
            cv2.rectangle(frame, (0, 0), (w, 65),
                          (60, 60, 60), -1)
            cv2.putText(frame,
                        "No person detected  move into frame",
                        (20, 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                        (100, 100, 255), 2, cv2.LINE_AA)
        cv2.imshow("Posture Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
print("Session ended.")