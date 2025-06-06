# apps/featurize/src/featurize/main.py

import typer
import structlog
from pathlib import Path
import torch
from torch import nn
from torchvision import models, transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import polars as pl
from tqdm import tqdm

app = typer.Typer()
log = structlog.get_logger()

@app.command()
def featurize(
    input_dir: Path = typer.Argument(..., help="Path to input directory (good/bad subfolders)."),
    output: Path = typer.Argument(..., help="Output Parquet file path."),
    batch_size: int = typer.Option(32, "--batch-size", "-b", help="Batch size."),
) -> None:
    """
    Featurize images → save as Parquet: image_path, label, features
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Loading ResNet18 backbone", device=str(device))

    # Model: ResNet18 backbone (no classifier)
    model = models.resnet18(weights="IMAGENET1K_V1")
    model.fc = nn.Identity()  # Removes classifier → outputs feature vector
    model = model.to(device)
    model.eval()

    # Transforms (same as train)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Dataset
    dataset = ImageFolder(root=str(input_dir), transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    log.info("Dataset loaded", num_images=len(dataset), classes=dataset.classes)

    # Featurize
    rows = {
        "image_path": [],
        "label": [],
        "features": [],
    }

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Featurizing"):
            images = images.to(device)
            features = model(images)  # shape: (batch, feature_dim)
            features = features.cpu().numpy()
            labels = labels.cpu().numpy()

            for i in range(features.shape[0]):
                image_path = dataset.samples[len(rows["image_path"])][0]
                label = labels[i]
                feature_vector = features[i].tolist()
                
                rows["image_path"].append(image_path)
                rows["label"].append(label)
                rows["features"].append(feature_vector)

    # Save to Parquet using Polars
    df = pl.DataFrame(rows)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(output)

    log.info("Featurization complete", output=str(output), num_rows=df.height)

if __name__ == "__main__":
    app()

