import base64
import io
from PIL import Image

import structlog

log = structlog.get_logger(__name__)


def decode_base64_image(base64_str: str) -> Image.Image:
    image_data = base64.b64decode(base64_str)
    return Image.open(io.BytesIO(image_data)).convert("RGB")


def encode_image_to_base64(image: Image.Image) -> str:
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def crop_box(image: Image.Image, box: list[float]) -> Image.Image:
    x1, y1, x2, y2 = box
    return image.crop((x1, y1, x2, y2))
