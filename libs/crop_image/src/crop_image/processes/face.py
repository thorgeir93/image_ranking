from pathlib import Path
from warnings import deprecated
from PIL import Image
import cv2
import dlib
import numpy as np
from typing import Literal
import structlog

log = structlog.get_logger(__name__)

# Load Haar Cascade (legacy method)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

repo_root = Path(__file__).resolve().parents[3]
models_dir = repo_root / 'models'

# Load DNN face detector (modern method)
dnn_model_path = {
    'prototxt': models_dir / "deploy.prototxt.txt",
    'weights': models_dir / "res10_300x300_ssd_iter_140000.caffemodel"
}
dnn_net = cv2.dnn.readNetFromCaffe(dnn_model_path['prototxt'], dnn_model_path['weights'])

face_detector = dlib.get_frontal_face_detector()

# TODO: remove ynet code if the dlib is doing well enough.
# Load YuNet ONNX model
# Make sure to update this path based on your project layout
# Create YuNet detector
# face_detector = cv2.FaceDetectorYN.create(
#     model=models_dir / "face_detection_yunet_2023mar.onnx",
#     config='',
#     input_size=(640, 640),
#     score_threshold=0.8,
#     nms_threshold=0.3,
#     top_k=5000,
#     backend_id=cv2.dnn.DNN_BACKEND_DEFAULT,
#     target_id=cv2.dnn.DNN_TARGET_CPU,
# )
# 
# def _has_face_yunet(img: np.ndarray) -> bool:
#     h, w = img.shape[:2]
#     face_detector.setInputSize((w, h))
#     _, faces = face_detector.detect(img)
#     return faces is not None and len(faces) > 0


def _has_face_haar(img: np.ndarray) -> bool:
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    return len(faces) > 0

def _has_face_dnn(img: np.ndarray, conf_threshold: float = 0.5) -> bool:
    h, w = img.shape[:2]
    blob = cv2.dnn.blobFromImage(img, 1.0, (300, 300), (104.0, 177.0, 123.0))
    dnn_net.setInput(blob)
    detections = dnn_net.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            return True
    return False

def _has_face_dlib(pil_image: Image.Image) -> bool:
    """
    Detects if there is a face in the given image using dlib's frontal face detector.

    Args:
        pil_image (PIL.Image.Image): Image to check for a face.

    Returns:
        bool: True if a face is detected, False otherwise.
    """
    img = np.array(pil_image.convert("RGB"))  # Convert PIL image to numpy array
    faces = face_detector(img)  # Detect faces
    return len(faces) > 0  # Return True if at least one face is detected

def has_face(pil_image: Image.Image, method: Literal["haar", "dnn", "dlib"] = "dlib") -> bool:
    """
    Determines if a PIL image contains at least one face.

    Args:
        pil_image (Image.Image): The input image as a PIL Image object.
        method (str): 'haar' for legacy cascade, 'dnn' for modern detector, or 'dlib' for dlib's frontal face detector. Default is 'dlib'.

    Returns:
        bool: True if the image contains at least one face, False otherwise.
    """
    img = np.array(pil_image)

    if method == "haar":
        log.debug("Using Haar Cascade method for face detection")
        return _has_face_haar(img)
    elif method == "dnn":
        log.debug("Using DNN method for face detection")
        return _has_face_dnn(img)
    elif method == "dlib":
        log.debug("Using dlib method for face detection")
        return _has_face_dlib(pil_image)
    else:
        raise ValueError("Invalid method. Choose 'haar', 'dnn', or 'dlib'.")

@deprecated("Not used anymore, use has_face() instead.")
def filter_images_with_faces(images: list) -> list:
    """
    Given: list of (image_path, img, *meta) OR (image_path, img)
    Returns: list of (image_path, img, faces_count)
    """
    images_with_faces = []

    for image_path, img, _, _ in images:
        # if len(image) >= 2:
        #     image_path, img = image[0], image[1]
        # else:
        #     log.error("Invalid image tuple", image=image)
        #     continue

        # image_path.save("temp.jpg")

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        faces_count = len(faces)

        if faces_count > 0:
            images_with_faces.append((image_path, img, faces_count))
            log.debug("Image has faces", path=str(image_path), faces_count=faces_count)
        else:
            pass
            log.debug("Image has no faces", path=str(image_path))

    log.info("Filtered images with faces", total=len(images_with_faces))
    return images_with_faces
