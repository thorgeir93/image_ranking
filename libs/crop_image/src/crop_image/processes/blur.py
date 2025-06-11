from typing import Any, List, Tuple
from pathlib import Path
import cv2
import numpy as np
from PIL import Image

import structlog

log = structlog.get_logger(__name__)


# TODO: add blurrness param to params.yaml in dvc
def is_blurred(img: Any, threshold: float = 100.0) -> bool:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    return variance < threshold


def blur_score(img: Any) -> float:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def is_sharp(pil_image: Image.Image, threshold: float = 100.0) -> bool:
    """
    Determines if a PIL image is sharp based on a blur threshold.

    Args:
        pil_image (Image.Image): The input image as a PIL Image object.
        threshold (float): The blur threshold. Images with a variance below this value are considered blurred.

    Returns:
        bool: True if the image is sharp, False otherwise.
    """
    # Convert PIL image to NumPy array
    img = np.array(pil_image)

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Calculate the variance of the Laplacian (blur score)
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()

    # Return True if the variance is above the threshold (sharp), otherwise False
    return variance >= threshold


def filter_sharp_images_from_images(
    images: List[Tuple[Path, np.ndarray, Any]], blur_threshold: float
) -> List[Tuple[Path, np.ndarray, float, bool]]:
    """
    Given: list of (image_path, img, _)
    Returns: list of (image_path, img, blur_score, is_blurred)
    """
    sharp_images = []

    # TODO: consider to move the blur functionality to here instead of in crop
    #       then we can skip _,_,_.
    for image_path, img, _ in images:
        score = blur_score(img)
        blurred = is_blurred(img, threshold=blur_threshold)

        if not blurred:
            sharp_images.append((image_path, img, score, blurred))
            log.debug("Image is sharp", path=str(image_path), blur_score=f"{score:.2f}")
        else:
            log.debug(
                "Image is blurred", path=str(image_path), blur_score=f"{score:.2f}"
            )

    log.info("Filtered sharp images", total=len(sharp_images))
    return sharp_images
