import numpy as np

def get_landmark_coords(landmarks, index, w, h):
    lm = landmarks[index]
    return [int(lm.x * w), int(lm.y * h)]

def calculate_angle(a, b, c):
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    c = np.array(c, dtype=float)
    ba = a - b
    bc = c - b
    cosine = np.dot(ba, bc) / (
        np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6
    )
    return round(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))), 2)

def calculate_neck_angle(landmarks, w, h):
    nose          = get_landmark_coords(landmarks, 0,  w, h)
    left_ear      = get_landmark_coords(landmarks, 7,  w, h)
    left_shoulder = get_landmark_coords(landmarks, 11, w, h)
    return calculate_angle(nose, left_ear, left_shoulder)

def calculate_shoulder_tilt(landmarks, w, h):
    left_shoulder  = get_landmark_coords(landmarks, 11, w, h)
    right_shoulder = get_landmark_coords(landmarks, 12, w, h)
    return abs(left_shoulder[1] - right_shoulder[1])

def is_bad_posture(neck_angle, shoulder_tilt,
                   neck_threshold=93, tilt_threshold=16):
    """
    Uses COMBINATION of neck angle + shoulder tilt.
    Bad posture = BOTH conditions triggered together.
    This prevents false alerts from either metric alone.
    """
    neck_bad  = neck_angle < neck_threshold
    tilt_bad  = shoulder_tilt > tilt_threshold

    reasons = []

    if neck_bad and tilt_bad:
        reasons.append(
            f"Bad posture — neck: {neck_angle} deg, tilt: {shoulder_tilt}px"
        )
    elif neck_bad:
        reasons.append(f"Head tilting forward: {neck_angle} deg")
    elif tilt_bad:
        reasons.append(f"Shoulders uneven: {shoulder_tilt}px")

    # Only truly bad if BOTH are triggered
    is_bad = neck_bad and tilt_bad

    return is_bad, reasons