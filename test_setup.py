import cv2
import mediapipe as mp
import numpy as np

print("OpenCV:", cv2.__version__)
print("MediaPipe:", mp.__version__)
print("NumPy:", np.__version__)

# Test webcam opens
cap = cv2.VideoCapture(0)
if cap.isOpened():
    print("Webcam: OK")
    cap.release()
else:
    print("Webcam: FAILED — check camera index")