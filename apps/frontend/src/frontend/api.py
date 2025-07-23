import httpx
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type(httpx.RequestError),
)
def submit_image(session: httpx.Client, image_b64: str, timeout: int = 10.0) -> dict:
    """
    Send the base64-encoded image to the rating API with retries.

    Args:
        session (httpx.Client): Reusable HTTP client.
        image_b64 (str): Base64-encoded image string.

    Returns:
        Parsed JSON response from the API.
    """
    response = session.post(
        url=session.base_url,
        json={"image": image_b64},
        timeout=timeout,
    )
    response.raise_for_status()
    return response.json()

def get_image_ranking(image_base64: str, api_url: str = "http://localhost:8000/image-ranking/running-style") -> dict:
    """
    Converts an image to Base64, sends it to the API, and retrieves the ranking.

    Args:
        image_base64: The base64-encoded image string.
        api_url: The URL of the API endpoint.

    Returns:
        dict: The API response containing the ranking.
    """
    # Prepare the payload
    payload = {
        "image_base64": image_base64
    }

    # Send the HTTP POST request 
    with httpx.Client() as client:
        response = client.post(api_url, json=payload)
        response.raise_for_status()

    # Return the JSON response
    return response.json()