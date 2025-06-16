from fastapi import FastAPI

# from app.api import router as ocr_router
from api.routes import router as image_ranking

app = FastAPI(title="Image Ranking API")

# Mount router
app.include_router(image_ranking, prefix="/image-ranking")