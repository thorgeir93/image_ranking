from pathlib import Path
import json

import numpy as np
import polars as pl
import torch
from torch import nn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import typer

app = typer.Typer()


def load_data(input_file: Path) -> tuple[list[list[float]], list[int], int]:
    """Load featurized evaluation data from Parquet using Polars."""
    df = pl.read_parquet(input_file)

    features_np = np.array(df["features"].to_list(), dtype=np.float32)
    feature_dim = features_np.shape[1]
    X = features_np.tolist()

    y = df["label"].to_list()

    return X, y, feature_dim


def build_model(feature_dim: int, dropout: float = 0.5) -> nn.Module:
    """Rebuild the exact same model structure used in training."""
    model = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(feature_dim, 1),
        nn.Sigmoid(),
    )
    return model

def load_model(model_path: Path, feature_dim: int, dropout: float = 0.5) -> nn.Module:
    """Rebuild model structure and load state_dict."""
    model = build_model(feature_dim, dropout)
    state_dict = torch.load(model_path, map_location=torch.device("cpu"))
    model.load_state_dict(state_dict)
    model.eval()
    return model

def evaluate_model(model: nn.Module, X: list[list[float]], y: list[int]) -> dict[str, float]:
    """Run evaluation and compute metrics."""
    X_tensor = torch.tensor(X, dtype=torch.float32)

    with torch.no_grad():
        outputs = model(X_tensor)
        y_pred_prob = outputs.squeeze().cpu().numpy()
        y_pred = (y_pred_prob >= 0.5).astype(int)

    metrics: dict[str, float] = {
        "accuracy": accuracy_score(y, y_pred),
        "precision": precision_score(y, y_pred, pos_label=1),
        "recall": recall_score(y, y_pred, pos_label=1),
        "f1": f1_score(y, y_pred, pos_label=1),
    }

    return metrics

def save_metrics(metrics: dict[str, float], metric_file: Path) -> None:
    """Save metrics as JSON."""
    metric_file.parent.mkdir(parents=True, exist_ok=True)
    with metric_file.open("w") as f:
        json.dump(metrics, f, indent=4)

@app.command()
def main(
    input_file: Path = typer.Argument(..., help="Path to featurized evaluation data (Parquet)"),
    model: Path = typer.Option(..., "--model", help="Path to trained model (state_dict .pth / .pkl)"),
    metric: Path = typer.Option(..., "--metric", help="Path to output metrics file (JSON)"),
    dropout: float = typer.Option(0.5, "--dropout", help="Dropout used in the model (should match training)"),
) -> None:
    """Evaluate PyTorch model on given dataset and save metrics."""
    typer.echo(f"Loading evaluation data from: {input_file}")

    X, y, feature_dim = load_data(input_file)
    typer.echo(f"Detected feature_dim={feature_dim}")

    model_instance = load_model(model, feature_dim, dropout)


    typer.echo(f"Loading model from: {model}")
    model_instance = load_model(model, feature_dim, dropout)

    typer.echo("Running evaluation...")
    metrics = evaluate_model(model_instance, X, y)

    typer.echo(f"Metrics: {metrics}")
    typer.echo(f"Saving metrics to: {metric}")
    save_metrics(metrics, metric)

    typer.echo("Evaluation completed.")

if __name__ == "__main__":
    app()
