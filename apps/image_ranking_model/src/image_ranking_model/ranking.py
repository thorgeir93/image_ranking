import base64
from PIL import Image
from io import BytesIO
import httpx

async def get_image_ranking(image: Image.Image, api_url: str = "http://localhost:8000/image-ranking/running-style") -> dict:
    """
    Converts an image to Base64, sends it to the API asynchronously, and retrieves the ranking.

    Args:
        image (Image.Image): The input image.
        api_url (str): The URL of the API endpoint.

    Returns:
        dict: The API response containing the ranking.
    """
    # Convert the image to Base64
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    image_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

    # Prepare the payload
    payload = {
        "image_base64": image_base64
    }

    # Send the HTTP POST request asynchronously
    async with httpx.AsyncClient() as client:
        response = await client.post(api_url, json=payload)
        response.raise_for_status()

    # Return the JSON response
    return response.json()
