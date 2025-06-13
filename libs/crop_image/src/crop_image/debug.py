from pathlib import Path
from PIL import Image
from datetime import datetime

import structlog

log = structlog.get_logger()


def save_debug_image(
    image: Image.Image, debug_dir: Path = Path("debug_images")
) -> Path:
    """
    Save an image to a centralized debug directory for troubleshooting.

    Args:
        image (Image.Image): The image to save.
        debug_dir (Path): The directory where debug images will be saved. Defaults to 'debug_images'.

    Returns:
        Path: The path to the saved debug image.
    """
    # Ensure the debug directory exists
    debug_dir.mkdir(parents=True, exist_ok=True)

    # Generate a timestamped filename for the debug image
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    debug_filename = f"debug_image_{timestamp}.png"
    debug_path = debug_dir / debug_filename

    # Save the image to the debug directory
    image.save(debug_path)

    log.debug("Debug image saved", path=str(debug_path))

    return debug_path
