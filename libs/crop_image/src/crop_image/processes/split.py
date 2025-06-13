from PIL import Image


def split_image(
    image: Image.Image, split_ratio: float = 0.5
) -> tuple[Image.Image, Image.Image]:
    """
    Splits an image into two parts based on the specified split ratio.

    Args:
        image (Image.Image): Input image as a PIL Image object.
        split_ratio (float): Ratio at which to split the image vertically.
                             Must be between 0 and 1. Default is 0.5 (equal halves).

    Returns:
        tuple[Image.Image, Image.Image]: Two images resulting from the split.
    """
    if not (0 < split_ratio < 1):
        raise ValueError("split_ratio must be between 0 and 1 (exclusive).")

    width, height = image.size
    split_height = int(height * split_ratio)

    # Crop the first part (top portion of the image)
    first_part = image.crop((0, 0, width, split_height))

    # Crop the second part (bottom portion of the image)
    second_part = image.crop((0, split_height, width, height))

    return first_part, second_part
