import cv2
import mediapipe as mp
import numpy as np

BaseOptions           = mp.tasks.BaseOptions
PoseLandmarker        = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode     = mp.tasks.vision.RunningMode

MODEL_PATH = "pose_landmarker.task"

def get_coords(lm, index, w, h):
    return [int(lm[index].x * w), int(lm[index].y * h)]

def dot_angle(a, b, c):
    """Angle at b using dot product — correct method"""
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    c = np.array(c, dtype=float)
    ba = a - b
    bc = c - b
    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return round(np.degrees(np.arccos(np.clip(cosine, -1, 1))), 2)

options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.IMAGE
)

cap = cv2.VideoCapture(0)

with PoseLandmarker.create_from_options(options) as landmarker:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]
        rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = landmarker.detect(mp_img)

        if result.pose_landmarks and len(result.pose_landmarks) > 0:
            lm = result.pose_landmarks[0]

            nose          = get_coords(lm, 0,  w, h)
            left_ear      = get_coords(lm, 7,  w, h)
            left_shoulder = get_coords(lm, 11, w, h)
            left_hip      = get_coords(lm, 23, w, h)

            # Two different angle methods side by side 
            # Method 1: nose -> ear -> shoulder (forward head tilt)
            angle_head = dot_angle(nose, left_ear, left_shoulder)

            # Method 2: ear -> shoulder -> hip (torso lean)
            angle_torso = dot_angle(left_ear, left_shoulder, left_hip)

            # Draw points 
            for pt, name, color in [
                (nose,          "nose",  (0,255,255)),
                (left_ear,      "ear",   (0,200,255)),
                (left_shoulder, "sh",    (255,100,0)),
                (left_hip,      "hip",   (100,255,100)),
            ]:
                cv2.circle(frame, tuple(pt), 7, color, -1)
                cv2.putText(frame, name, (pt[0]+8, pt[1]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            # Draw angle lines 
            cv2.line(frame, tuple(nose),     tuple(left_ear),      (0,200,255), 2)
            cv2.line(frame, tuple(left_ear), tuple(left_shoulder),  (0,200,255), 2)
            cv2.line(frame, tuple(left_ear), tuple(left_shoulder),  (255,100,0), 2)
            cv2.line(frame, tuple(left_shoulder), tuple(left_hip),  (255,100,0), 2)

            # Print to terminal
            print(f"Head angle (nose→ear→sh): {angle_head}  |  "
                  f"Torso angle (ear→sh→hip): {angle_torso}")

            # Show on frame 
            cv2.rectangle(frame, (0,0), (w, 70), (30,30,30), -1)
            cv2.putText(frame,
                f"Head tilt (nose->ear->sh): {angle_head} deg",
                (20, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (0,200,255), 2)
            cv2.putText(frame,
                f"Torso lean (ear->sh->hip): {angle_torso} deg",
                (20, 58), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (255,150,0), 2)

        cv2.imshow("Debug Angles", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()