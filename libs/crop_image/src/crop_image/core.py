from calendar import c
from pathlib import Path
from typing import Callable
from PIL import Image
import structlog
from ultralytics import YOLO

from crop_image.constants import YOLO_MODEL, YOLO_MODEL_BASE_PATH
from crop_image.models import ImageParts
from crop_image.processes.blur import is_sharp
from crop_image.processes.crop import crop_person
from crop_image.processes.face import has_face
from crop_image.processes.split import split_image  # Import the generic split_image method

logger = structlog.get_logger()



def pipeline(
    image: Image.Image,
    yolo_model: str = YOLO_MODEL,
    crop_min_confidence: float = 0.5,
    crop_person_fn: Callable[
        [Image.Image, YOLO, float], list[Image.Image]
    ] = crop_person,
    is_sharp_fn: Callable[[Image.Image], bool] = is_sharp,
    has_face_fn: Callable[[Image.Image], bool] = has_face,
    split_image_fn: Callable[[Image.Image, float], tuple[Image.Image, Image.Image]] = split_image,
    split_ratio: float = 0.5,
) -> list[ImageParts]:
    """
    Process an image through the pipeline:
    1. Crop all detected persons from the image.
    2. Filter only sharp images.
    3. Accept only images that contain a face.
    4. Split the image into two parts based on the split ratio.

Args:
        image (Image.Image): Input image as a PIL Image object.
        crop_person_fn (Callable): Function to crop persons from the image.
        is_sharp_fn (Callable): Function to check if an image is sharp.
        has_face_fn (Callable): Function to check if an image contains a face.
        split_image_fn (Callable): Function to split an image into two parts.
        split_ratio (float): Ratio at which to split the image vertically.

    Returns:
        list[ImageParts]: List of ImageParts containing upper and lower body images.
    """
    logger.debug("Starting pipeline", step="initial", image_info=str(image))

    crop_model = YOLO(YOLO_MODEL_BASE_PATH / yolo_model)

    # Step 1: Crop all detected persons from the image
    cropped_images: list[Image.Image] = crop_person_fn(
        image, crop_model, crop_min_confidence
    )
    if not cropped_images:
        logger.warning("No persons detected in the image", step="crop_person")
        return []

    logger.debug(
        "Cropped persons detected", step="crop_person", count=len(cropped_images)
    )

    processed_images: list[ImageParts] = []

    for idx, cropped_image in enumerate(cropped_images):
        logger.debug("Processing cropped image", step="process_cropped", index=idx)

        # Step 2: Filter only sharp images
        if not is_sharp_fn(cropped_image):
            logger.warning("Image is not sharp", step="is_sharp", index=idx)
            continue

        # Step 3: Accept only images that contain a face
        if not has_face_fn(cropped_image):
            logger.warning("No face detected in the image", step="has_face", index=idx)
            continue

        # Step 4: Split the image into two parts
        try:
            part1, part2 = split_image_fn(cropped_image, split_ratio)
            logger.debug("Image split into two parts", step="split_image", index=idx)
            processed_images.append(ImageParts(upper_body=part1, lower_body=part2))
        except Exception as e:
            logger.error("Failed to split image", step="split_image", index=idx, error=str(e))
            continue

    logger.debug(
        "Pipeline completed", step="completed", processed_count=len(processed_images)
    )
    return processed_images


def process_image_pipeline(
    image_path: str,
    yolo_model: str = YOLO_MODEL,
    crop_min_confidence: float = 0.5,
    output_dir: Path = Path("."),
):
    """
    Process an image using the YOLO model and save the cropped results.

    Args:
        image_path (str): Path to the input image.
        yolo_model (str): Path to the YOLO model file. Defaults to YOLO_MODEL.
        crop_min_confidence (float): Minimum confidence for cropping persons. Defaults to 0.5.
        output_dir (Path): Directory to save the processed images. Defaults to the current directory.

    Returns:
        List[str]: List of file paths to the saved processed images.
    """
    image = Image.open(image_path)
    processed_images = pipeline(image, yolo_model, crop_min_confidence)

    saved_paths = []
    output_dir.mkdir(parents=True, exist_ok=True)

    # Extract the original image's name (without extension) and extension
    original_name = Path(image_path).stem
    original_extension = Path(image_path).suffix

    for idx, image_parts in enumerate(processed_images):
        # Save the upper body image
        upper_output_path = output_dir / f"{original_name}_processed_upper_{idx}{original_extension}"
        image_parts.upper_body.save(upper_output_path)
        saved_paths.append(str(upper_output_path))

        # Save the lower body image
        lower_output_path = output_dir / f"{original_name}_processed_lower_{idx}{original_extension}"
        image_parts.lower_body.save(lower_output_path)
        saved_paths.append(str(lower_output_path))

    return saved_paths
