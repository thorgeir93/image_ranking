from pydantic import BaseModel
from PIL import Image

class APIRequest(BaseModel):
    image_base64: str  # Base64 encoded image

class Ranking(BaseModel):
    # The image being ranked, encoded in Base64
    image_base64: str
    ranking: float | None

class ImagePartRanking(BaseModel):
    # The original image in Base64 format
    image_base64: str
    lower_body: Ranking

class APIResponse(BaseModel):
    results: list[ImagePartRanking]