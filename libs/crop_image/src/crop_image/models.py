from PIL import Image
from pydantic import BaseModel


class ImageParts(BaseModel):
    upper_body: Image.Image
    lower_body: Image.Image

    class Config:
        arbitrary_types_allowed = True
