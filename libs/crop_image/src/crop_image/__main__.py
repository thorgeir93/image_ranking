from typing import Final
import typer
from PIL import Image
from ultralytics import YOLO
from crop_image.constants import YOLO_MODEL, YOLO_MODEL_BASE_PATH
from crop_image.pipeline import pipeline, process_image_pipeline
from pathlib import Path

app = typer.Typer()

@app.command()
def run_pipeline(
    image_path: str = typer.Argument(..., help="Path to the input image."),
    yolo_model: str = typer.Option(
        YOLO_MODEL, help=f"Path to the YOLO model file. Defaults to '{YOLO_MODEL}'."
    ),
    crop_min_confidence: float = typer.Option(
        0.5, help="Minimum confidence for cropping persons. Defaults to 0.5."
    ),
    output_dir: Path = typer.Option(
        Path("."), help="Directory to save the processed images. Defaults to the current directory."
    ),
):
    """
    Run the image processing pipeline.

    Args:
        image_path (str): Path to the input image.
        model_path (str): Path to the YOLO model file.
        crop_min_confidence (float): Minimum confidence for cropping persons.
        output_dir (str): Directory to save the processed images.
    """
    saved_paths = process_image_pipeline(image_path, yolo_model, crop_min_confidence, output_dir)
    for path in saved_paths:
        typer.echo(f"Saved processed image: {path}")


if __name__ == "__main__":
    app()
