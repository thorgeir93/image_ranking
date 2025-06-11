from tabnanny import verbose
from typing import Any, List, Tuple, Union
import cv2
from pathlib import Path
from ultralytics import YOLO
from PIL import Image
import numpy as np

from crop_image.utils import is_image_file
from crop_image.processes.blur import is_blurred, blur_score

import structlog

log = structlog.get_logger(__name__)


def load_yolo_model(model_path: Path = Path("yolov8n.pt")) -> YOLO:
    """
    Returns: YOLO model instance
    """
    if getattr(load_yolo_model, "__model", None) is not None:
        return load_yolo_model.__model
    else:
        load_yolo_model.__model = YOLO(model_path)
        return load_yolo_model.__model


def get_cropped_persons(
    img: Any, model: YOLO, min_confidence: float
) -> List[Tuple[np.ndarray, float]]:
    """
    Returns: List of tuples containing:
        - cropped_person_img: The cropped person image as a NumPy array
        - confidence: The confidence score of the detection
    """
    results = model(img, verbose=False)
    cropped_persons = []

    for result in results:
        for box in result.boxes:
            cls_id = int(box.cls[0])
            label = result.names[cls_id]
            confidence = float(box.conf[0])
            if label == "person" and confidence >= min_confidence:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cropped = img[y1:y2, x1:x2]
                cropped_persons.append((cropped, confidence))

    return cropped_persons


def crop_persons_from_image(
    image_path: Path, model: YOLO, min_confidence: float
) -> List[Tuple[Path, np.ndarray, float]]:
    """
    Returns: List of tuples containing:
        - image_path: The path to the original image
        - cropped_person_img: The cropped person image as a NumPy array
        - confidence: The confidence score of the detection
    """
    img = cv2.imread(str(image_path))
    if img is None:
        log.error("Failed to load image", path=str(image_path))
        return []

    cropped_persons_raw = get_cropped_persons(img, model, min_confidence)
    cropped_persons = []

    for cropped, confidence in cropped_persons_raw:
        cropped_persons.append((image_path, cropped, confidence))

    return cropped_persons


def get_cropped_persons_from_directory(
    source_dir: Path, model: YOLO, min_confidence: float
) -> List[Tuple[Path, np.ndarray, float]]:
    """
    Returns: List of tuples containing:
        - image_path: The path to the original image
        - cropped_person_img: The cropped person image as a NumPy array
        - confidence: The confidence score of the detection
    """
    image_paths = [p for p in source_dir.iterdir() if p.is_file() and is_image_file(p)]
    if not image_paths:
        log.warning(
            "No image files found in source directory", source_dir=str(source_dir)
        )
        return []

    log.info(
        "Found images to process for person cropping",
        count=len(image_paths),
        source_dir=str(source_dir),
    )

    all_cropped_persons = []

    for image_path in image_paths:
        cropped_persons = crop_persons_from_image(image_path, model, min_confidence)
        all_cropped_persons.extend(cropped_persons)

    log.info("Total cropped persons", total=len(all_cropped_persons))
    return all_cropped_persons


def crop_persons_from_pil_image(
    pil_image: Image.Image, model: YOLO, min_confidence: float
) -> List[Tuple[Image.Image, np.ndarray, float]]:
    """
    Returns: List of tuples containing:
        - pil_image: The original PIL image
        - cropped_person_img: The cropped person image as a NumPy array
        - confidence: The confidence score of the detection
    """
    img = np.array(pil_image)  # Convert PIL image to NumPy array
    if img is None:
        log.error("Failed to process PIL image")
        return []

    cropped_persons_raw = get_cropped_persons(img, model, min_confidence)
    cropped_persons = []

    for cropped, confidence in cropped_persons_raw:
        cropped_persons.append((pil_image, cropped, confidence))

    return cropped_persons


def crop_person(
    pil_image: Image.Image, model: YOLO, min_confidence: float
) -> list[Image.Image]:
    """
    Crops all detected persons from a PIL image.

    Args:
        pil_image (Image.Image): The input PIL image.
        model (YOLO): The YOLO model used for person detection.
        min_confidence (float): Minimum confidence threshold for detections.

    Returns:
        list[Image.Image]: List of cropped PIL images containing detected persons.
    """
    cropped_persons_raw = crop_persons_from_pil_image(pil_image, model, min_confidence)
    cropped_images: list[Image.Image] = []

    for _, cropped_np, _ in cropped_persons_raw:
        # Convert NumPy array back to PIL Image
        cropped_pil = Image.fromarray(cropped_np)
        cropped_images.append(cropped_pil)

    return cropped_images
