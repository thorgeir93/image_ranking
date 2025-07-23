import httpx
from typing import Dict
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type(httpx.RequestError),
)
async def submit_image(session: httpx.AsyncClient, image_b64: str, timeout: int = 10.0) -> dict:
    """
    Send the base64-encoded image to the rating API with retries.

    Args:
        session (httpx.AsyncClient): Reusable HTTP client.
        image_b64 (str): Base64-encoded image string.

    Returns:
        Dict: Parsed JSON response from the API.
    """
    response = await session.post(
        url=session.base_url,
        json={"image": image_b64},
        timeout=timeout,
    )
    response.raise_for_status()
    return response.json()