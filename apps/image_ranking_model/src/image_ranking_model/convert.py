from pathlib import Path
import io

import rawpy
from PIL import Image


def convert_raw_to_jpg(path: Path, quality: int = 75, max_dim: int = 2048) -> bytes:
    """
    Convert a RAW image file (e.g., .CR2, .NEF) to a JPEG byte stream.

    Args:
        path: Path to the RAW image file.
        qualit: JPEG quality (1–100). Default is 85.
        max_dim: Maximum dimension for resizing the image while keeping aspect ratio.

    Returns:
        Optional[bytes]: JPEG-encoded image as bytes if successful, None on failure.
    """
    # Load and postprocess RAW image
    with rawpy.imread(str(path)) as raw:
        rgb_image = raw.postprocess(use_camera_wb=True, no_auto_bright=True)

    # Resize if needed while keeping aspect ratio
    img.thumbnail((max_dim, max_dim), resample=Image.LANCZOS)

    # Convert numpy array to PIL Image and encode to JPG in-memory
    img = Image.fromarray(rgb_image)
    with io.BytesIO() as output_buffer:
        img.save(output_buffer, format="JPEG", quality=quality)
        return output_buffer.getvalue()
