# ============================================
# Face to Desmos Generator
# Description: Extracts facial landmarks using Mediapipe and
#              generates line segments to plot in Desmos.
# ============================================

import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
import os

# ============================================
# Configuration
# ============================================

CANVAS_WIDTH = 500
CANVAS_HEIGHT = 600

# ============================================
# Utility Functions
# ============================================

def generate_desmos_line(equation, x_start, x_end):
    return f"{equation} {{x \u2265 {x_start:.3f} and x \u2264 {x_end:.3f}}}"

def calculate_line(x1, y1, x2, y2):
    if x1 == x2:
        return None  # Vertical line, not representable as y = mx + b
    slope = (y2 - y1) / (x2 - x1)
    intercept = y1 - slope * x1
    return slope, intercept

def save_desmos_file(x_coords, y_coords, output_filename):
    lines = []
    for i in range(len(x_coords) - 1):
        p1 = (x_coords[i], y_coords[i])
        p2 = (x_coords[i + 1], y_coords[i + 1])
        line = calculate_line(*p1, *p2)
        if line:
            m, b = line
            lines.append(generate_desmos_line(f"y = {m:.3f}x + {b:.3f}", p1[0], p2[0]))

    with open(output_filename, "w") as file:
        for line in lines:
            file.write(line + "\n")
    print(f"Desmos expressions exported to '{output_filename}'")

# ============================================
# Face Landmark Detection
# ============================================

def extract_face_landmarks(image_path):
    face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=True)
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(image_rgb)

    if not result.multi_face_landmarks:
        raise ValueError("No face detected in the provided image.")

    height, width = image.shape[:2]
    landmarks = result.multi_face_landmarks[0]

    x_coords = [lm.x * width for lm in landmarks.landmark]
    y_coords = [lm.y * height for lm in landmarks.landmark]

    return np.array(x_coords), np.array(y_coords)

# ============================================
# Visualization
# ============================================

def visualize_landmarks(x_coords, y_coords):
    plt.figure(figsize=(8, 10))
    plt.scatter(x_coords, y_coords, s=2)
    plt.gca().invert_yaxis()
    plt.title("Detected Face Landmarks")
    plt.axis("off")
    plt.show()

# ============================================
# Main Execution
# ============================================

def process_image(image_path):
    print(f"Processing: {image_path}")

    x_coords, y_coords = extract_face_landmarks(image_path)

    # Define landmark groups
    regions = {
        "jawline": list(range(0, 17)),
        "left_eye": list(range(33, 42)),
        "right_eye": list(range(263, 272)),
        "lips": list(range(61, 80)),
    }

    visualize_landmarks(x_coords, y_coords)

    # Export selected regions to Desmos format
    for region_name, indices in regions.items():
        output_file = f"desmos_{region_name}.txt"
        save_desmos_file(x_coords[indices], y_coords[indices], output_file)

# ============================================
# Entry Point
# ============================================

if __name__ == "__main__":
    # TODO: Replace with your image path
    image_path = "your_image.jpg"
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    process_image(image_path)
