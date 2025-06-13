from pathlib import Path
from crop_image.utils import get_model
import dlib
import cv2
import numpy as np
from PIL import Image

# ***********+ UNFINISHED CODE ************
# Created to try to be more narrow when
# detecting a face on a image. But dlib is
# doing well enough for now.
# -----------------------------------------

# Load detector and predictor
predictor_path: Path = get_model(
    "shape_predictor_68_face_landmarks.dat"
)  # Adjust if needed
face_detector = dlib.get_frontal_face_detector()
landmark_predictor = dlib.shape_predictor(str(predictor_path))


def has_glasses(pil_image: Image.Image) -> bool:
    """
    Detects the presence of glasses on a face in the given image.

    Args:
        pil_image (PIL.Image.Image): Image containing a face.

    Returns:
        bool: True if glasses are detected, False otherwise.
    """
    img = np.array(pil_image.convert("RGB"))

    # Detect face
    faces = face_detector(img)
    if len(faces) == 0:
        return False  # No face detected

    rect = faces[0]
    shape = landmark_predictor(img, rect)
    landmarks = np.array([[pt.x, pt.y] for pt in shape.parts()])

    # Extract relevant landmark indices for the nose bridge
    nose_bridge_x = [landmarks[i][0] for i in [28, 29, 30, 31, 33, 34, 35]]
    nose_bridge_y = [landmarks[i][1] for i in [28, 29, 30, 31, 33, 34, 35]]

    x_min = min(nose_bridge_x)
    x_max = max(nose_bridge_x)
    y_min = landmarks[20][1]  # top of eyebrows
    y_max = landmarks[31][1]  # bottom of nose bridge

    # Crop region over the nose bridge
    cropped = pil_image.crop((x_min, y_min, x_max, y_max))
    cropped_np = np.array(cropped)

    # Apply Gaussian Blur and Canny edge detection
    blurred = cv2.GaussianBlur(cropped_np, (3, 3), sigmaX=0, sigmaY=0)
    edges = cv2.Canny(blurred, threshold1=100, threshold2=200)

    # Get vertical center column of edges
    center_col = edges[:, edges.shape[1] // 2]

    # If we find edge pixels (value = 255) in the center strip → glasses likely present
    return 255 in center_col
