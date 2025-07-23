import base64


def encode_to_base64(jpg_bytes: bytes) -> str:
    """
    Encode JPEG bytes into a base64 string.

    Args:
        jpg_bytes (bytes): JPEG image data.

    Returns:
        str: Base64-encoded string.
    """
    return base64.b64encode(jpg_bytes).decode("utf-8")
