from pathlib import Path
from PIL import Image
from cv2 import blur
from sympy import per

# from api.detector import load_detector, run_detection
# from api.model import load_model, run_model
from api.models import ImagePartRanking
from api.utils import crop_box, decode_base64_image, encode_image_to_base64
from api.onnx_model import ONNXClassifier, get_model_path

from crop_image.core import pipeline
from crop_image.models import ImageParts

import structlog

log = structlog.get_logger(__name__)


def process_image_pipeline(
    image_base64: str, person_crop_confidence: float = 0.5, sharpness_threshold: float = 100.0

) -> ImagePartRanking:

    # TODO: use version when possible
    input_image: Image.Image = decode_base64_image(image_base64)

    images_cropped: list[ImageParts] = pipeline(image=input_image, person_crop_confidence=person_crop_confidence, sharpness_threshold=sharpness_threshold)

    model_path: Path = get_model_path("lower_body.onnx")
    onnx_model = ONNXClassifier(model_path=model_path)

    results: list[dict] = []

    # TODO run model on lower images
    log.info("STEP 5: Run model on lower body images")
    for image_parts in images_cropped:
        log.info("Running model on lower body image")
        res = onnx_model.classify_image(image_parts.lower_body)
        log.info("Model results", results=res)
        results.append(
            ImagePartRanking(
                # image_base64=encode_image_to_base64(input_image),
                lower_body={
                    "image_base64": encode_image_to_base64(image_parts.lower_body),
                    "ranking": res
                }
            ) 
        )
    
    return results
