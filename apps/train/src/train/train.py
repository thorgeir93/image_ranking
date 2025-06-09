import json
import numpy as np
import structlog
from pathlib import Path
import typer
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import polars as pl

app = typer.Typer()
log = structlog.get_logger()


@app.command()
def train(
    parquet_path: Path = typer.Argument(
        ...,
        help="Path to Parquet file with featurized data (from featurize step).",
    ),
    epochs: int = typer.Option(
        10,
        "--epochs",
        "-e",
        help="Number of training epochs.",
    ),
    batch_size: int = typer.Option(
        32,
        "--batch-size",
        "-b",
        help="Batch size.",
    ),
    lr: float = typer.Option(
        1e-3,
        "--lr",
        "-l",
        help="Learning rate.",
    ),
    dropout: float = typer.Option(
        0.5,
        "--dropout",
        "-d",
        help="Dropout probability.",
    ),
    output_model: Path = typer.Option(
        "model/style_head_lower_body_01.pth",
        "--output-model",
        "-o",
        help="Path to save trained model.",
    ),
    metric_path: Path = typer.Option(
        None,
        "--metric",
        "-m",
        help="Path to save metrics (JSON).",
    ),
) -> None:
    """
    Train classification head on featurized image data.
    Outputs score 0.0 - 1.0.
    """

    # Log parameters
    log.info(
        "Training parameters",
        parquet_path=str(parquet_path),
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        dropout=dropout,
        output_model=str(output_model),
    )

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Using device", device=str(device))

    # Load featurized data
    df = pl.read_parquet(parquet_path)
    log.info("Parquet loaded", num_rows=df.height)

    # # Prepare tensors
    # features = torch.tensor(df["features"].to_numpy(), dtype=torch.float32)
    # labels = torch.tensor(df["label"].to_numpy(), dtype=torch.float32).unsqueeze(1)

    # log.info("Feature tensor", shape=tuple(features.shape))
    # log.info("Label tensor", shape=tuple(labels.shape))

    # Prepare tensors
    features_np = np.array(df["features"].to_list(), dtype=np.float32)
    features = torch.tensor(features_np, dtype=torch.float32)

    labels = torch.tensor(df["label"].to_numpy(), dtype=torch.float32).unsqueeze(1)

    log.info("Feature tensor", shape=tuple(features.shape))
    log.info("Label tensor", shape=tuple(labels.shape))

    # Create dataset + loader
    dataset = TensorDataset(features, labels)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Define simple classifier head
    feature_dim = features.shape[1]
    model = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(feature_dim, 1),
        nn.Sigmoid(),  # Output score 0.0 - 1.0
    )
    model = model.to(device)

    # Optimizer, loss
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()

    # Training loop
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for batch_features, batch_labels in loader:
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)

            optimizer.zero_grad()
            outputs = model(batch_features)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        log.info("Epoch complete", epoch=epoch + 1, avg_loss=avg_loss)

    # Save model
    output_model.parent.mkdir(exist_ok=True, parents=True)
    torch.save(model.state_dict(), output_model)
    log.info("Training complete", model_path=str(output_model))

    # Save metric if requested
    if metric_path:
        metric_data = {
            "avg_loss": avg_loss,
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "dropout": dropout,
        }
        metric_path.parent.mkdir(parents=True, exist_ok=True)
        with open(metric_path, "w") as f:
            json.dump(metric_data, f, indent=2)
        log.info("Saved metrics", path=str(metric_path))


if __name__ == "__main__":
    app()
