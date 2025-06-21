from pathlib import Path
from torchvision import transforms
from torchvision.models import resnet18
import torch
import numpy as np
from PIL import Image
import onnxruntime as ort

import structlog

log = structlog.get_logger(__name__)


def get_model_path(model_name: str) -> Path:
    """Return the path to the given model name."""
    project_root_path = Path(__file__).resolve().parents[4]
    models_dir = project_root_path / "exported_models"
    model_path = models_dir / model_name
    if not model_path.exists():
        log.error("Model not found", model=str(model_path))
        raise FileNotFoundError(f"Model not found: {model_path}")
    return model_path.resolve()


class ONNXClassifier:
    def __init__(self, model_path: str):
        self.session = ort.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name
        self.feature_extractor = resnet18(pretrained=True)
        self.feature_extractor.fc = torch.nn.Identity()  # Remove classification head
        self.feature_extractor.eval()

        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def extract_features(self, image: Image.Image) -> np.ndarray:
        tensor = self.transform(image).unsqueeze(0)  # Shape: (1, 3, 224, 224)
        with torch.no_grad():
            features = self.feature_extractor(tensor)  # Shape: (1, 512)
        return features.numpy()

    def classify_image(self, image: Image.Image) -> float:
        feature_vector = self.extract_features(image)
        result = self.session.run(
            None, {self.input_name: feature_vector.astype(np.float32)}
        )
        return float(result[0][0])
