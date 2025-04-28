# ===============================
# Face to Desmos Generator
# Notes: Uses Mediapipe for face detection
# ===============================

import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
import os

# ===============================
# Configuration
# ===============================

CANVAS_WIDTH = 500
CANVAS_HEIGHT = 600

# ===============================
# Helper Functions
# ===============================

def desmos_line(y_func, x_min, x_max):
    return f"{y_func} {{x >= {x_min:.3f} and x <= {x_max:.3f}}}"

def fit_line(x1, y1, x2, y2):
    if x2 - x1 == 0:
        return None
    slope = (y2 - y1) / (x2 - x1)
    intercept = y1 - slope * x1
    return slope, intercept

def export_to_desmos(x_points, y_points, filename="desmos_real_face.txt"):
    lines = []
    for i in range(len(x_points) - 1):
        x1, y1 = x_points[i], y_points[i]
        x2, y2 = x_points[i+1], y_points[i+1]
        fitted = fit_line(x1, y1, x2, y2)
        if fitted:
            slope, intercept = fitted
            lines.append(desmos_line(f"y = {slope:.3f}x + {intercept:.3f}", x1, x2))
    with open(filename, "w") as f:
        for line in lines:
            f.write(line + "\n")
    print(f"Exported Desmos lines to {filename}")

# ===============================
# Face Detection Function
# ===============================

def detect_face_landmarks(image_path):
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True)

    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(img_rgb)

    height, width, _ = img.shape
    if not results.multi_face_landmarks:
        raise ValueError("No face detected!")

    landmarks = results.multi_face_landmarks[0]

    x_points = []
    y_points = []
    for lm in landmarks.landmark:
        x = lm.x * width
        y = lm.y * height
        x_points.append(x)
        y_points.append(y)

    return np.array(x_points), np.array(y_points)

# ===============================
# Plot Function
# ===============================

def plot_face(x_points, y_points):
    plt.figure(figsize=(8, 10))
    plt.scatter(x_points, y_points, s=1)
    plt.gca().invert_yaxis()
    plt.title("Detected Facial Landmarks")
    plt.show()

# ===============================
# Main Runner
# ===============================

def main(image_path):
    print(f"Processing {image_path}")
    x_points, y_points = detect_face_landmarks(image_path)

    jawline_idx = list(range(0, 17))
    left_eye_idx = list(range(33, 42))
    right_eye_idx = list(range(263, 272))
    lips_idx = list(range(61, 80))

    plot_face(x_points, y_points)

    export_to_desmos(x_points[jawline_idx], y_points[jawline_idx], filename="desmos_jawline.txt")
    export_to_desmos(x_points[left_eye_idx], y_points[left_eye_idx], filename="desmos_left_eye.txt")
    export_to_desmos(x_points[right_eye_idx], y_points[right_eye_idx], filename="desmos_right_eye.txt")
    export_to_desmos(x_points[lips_idx], y_points[lips_idx], filename="desmos_lips.txt")

# ===============================
# Run
# ===============================

if __name__ == "__main__":
    # <<< CHANGE THIS >>>
    image_path = "your_image_path.jpg"
    main(image_path)
