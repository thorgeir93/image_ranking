from fastapi import APIRouter, Query
from api.models import APIRequest, APIResponse
from api.utils import decode_base64_image
from api.service import process_image_pipeline

router = APIRouter()

# TODO: create endpoint to preprocess a image.

@router.post("/running-style", response_model=APIResponse)
def image_ranking(
    request: APIRequest,
    person_crop_confidence: float = Query(
        0.7, 
        description=(
            "Minimum confidence threshold for cropping a person from the image using a model. "
            "Detections with confidence scores below this value will be ignored."
        )
    ),
    sharpness_threshold: float = Query(
        100.0,
        description=(
            "Sharpness threshold for determining if an image is sharp. "
            "Images with a variance below this value are considered blurred."
        )
    )
) -> APIResponse:
    results = process_image_pipeline(
        image_base64=request.image_base64,
        person_crop_confidence=person_crop_confidence,
        sharpness_threshold=sharpness_threshold
    )
    
    print(f"Results: {results}")
    return APIResponse(results=results)

