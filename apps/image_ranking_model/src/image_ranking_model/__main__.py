import asyncio
import base64
from pathlib import Path
from typing import List
import os
import typer
from PIL import Image
from image_ranking_model.load_image import load_cr3_as_image
from image_ranking_model.ranking import get_image_ranking

import structlog

log = structlog.get_logger(__name__)

app = typer.Typer(help="Image ranking application.")

async def process_image(image: Image.Image) -> float:
    """
    Processes an image and returns its ranking based on running style.

    Args:
        image (Image.Image): The input image.

    Returns:
        float: The ranking score of the image.
    """
    ranking: dict = await get_image_ranking(image)

    if results := ranking.get("results", []):
        for result in results:
            if lower_body := result.get("lower_body", {}):
                if image_base64 := lower_body.get("image_base64"):
                    random_filename = f"ranked_image_{os.urandom(4).hex()}.jpg"
                    # Save the image asynchronously
                    await asyncio.to_thread(
                        lambda: open(random_filename, "wb").write(base64.b64decode(image_base64))
                    )
                    log.info("Saved ranked image for inspection", filename=random_filename)
                return result["lower_body"]["ranking"]
    return 0.0

@app.command()
def rank_images(
    input_dir: Path = typer.Argument(..., help="Path to directory with RAW images."),
    batch_size: int = typer.Option(10, "--batch-size", "-b", help="Batch size."),
) -> None:
    """
    Loads RAW images from a directory, processes them in batches, and ranks them.

    Args:
        input_dir (Path): Directory containing RAW images.
        batch_size (int): Number of images to process in a batch.
    """
    asyncio.run(async_rank_images(input_dir, batch_size))

async def async_rank_images(input_dir: Path, batch_size: int) -> None:
    """
    Asynchronous implementation of the rank_images function.

    Args:
        input_dir (Path): Directory containing RAW images.
        batch_size (int): Number of images to process in a batch.
    """
    input_dir = input_dir.resolve()
    if not input_dir.is_dir():
        log.error("Input path is not a directory", path=str(input_dir))
        raise typer.Exit(code=1)

    raw_files = sorted(input_dir.glob("*.CR3"), key=lambda x: x.name)  # Sort files by filename
    if not raw_files:
        log.error("No RAW (.CR3) files found in directory", path=str(input_dir))
        raise typer.Exit(code=1)

    log.info("Found RAW images", count=len(raw_files), batch_size=batch_size)

    for i in range(0, len(raw_files), batch_size):
        batch_files = raw_files[i:i + batch_size]
        log.info("Processing batch", batch_number=i // batch_size + 1)

        # Load images asynchronously
        images = await asyncio.gather(*[asyncio.to_thread(load_cr3_as_image, file) for file in batch_files])

        # Save the first image in the batch to disk for manual inspection
        if images:
            temp_image_path = f"temp_image_batch_{i // batch_size + 1}.jpg"
            await asyncio.to_thread(images[0].save, temp_image_path)
            log.info("Saved temporary image for inspection", path=temp_image_path)

        # Process images asynchronously
        rankings = await asyncio.gather(*[process_image(image) for image in images])

        for file, ranking in zip(batch_files, rankings):
            log.info("Image ranked", filename=file.name, ranking=ranking)

if __name__ == "__main__":
    app()