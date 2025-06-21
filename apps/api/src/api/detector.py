from ultralytics import YOLO
import numpy as np
from PIL import Image

_detector_cache: dict[str, YOLO] = {}


def load_detector(model_name: str, version: str) -> YOLO:
    # TODO: use version if possible
    global _detector_cache
    key = f"{model_name}-{version}"
    if key not in _detector_cache:
        # model_path = f"models/{model_name}/{version}/weights.pt"  # Example path
        model_path = "yolov8n.pt"  # Example pre-trained model
        _detector_cache[key] = YOLO(model_path)
    return _detector_cache[key]


def run_detection(
    detector: YOLO, image: Image.Image
) -> list[tuple[list[float], float]]:
    # Convert to np.array
    img_np = np.array(image)

    # Run inference
    results = detector.predict(img_np, conf=0.5)
    bboxes = []
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            score = box.conf[0].item()
            bboxes.append(([x1, y1, x2, y2], score))
    return bboxes
