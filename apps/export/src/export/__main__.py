import typer
import torch
import structlog
from pathlib import Path
import torch.nn as nn

app = typer.Typer()
log = structlog.get_logger()


def build_model(input_dim: int, dropout: float) -> nn.Module:
    return nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(input_dim, 1),
        nn.Sigmoid(),
    )


@app.command()
def export_model(
    input_model: Path = typer.Argument(..., help="Path to trained model .pth (state_dict)."),
    output_model: Path = typer.Argument(..., help="Path to save exported ONNX model."),
    format: str = typer.Option("onnx", "--format", "-f", help="Export format (default: onnx)."),
    input_dim: int = typer.Option(1024, "--input-dim", help="Input feature size."),  # Adjust as needed
    dropout: float = typer.Option(0.5, "--dropout", help="Dropout value to match training."),
) -> None:
    """
    Export a trained PyTorch model to ONNX format.
    """
    log.info("Loading model", input_model=str(input_model))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(input_dim=input_dim, dropout=dropout).to(device)
    model.load_state_dict(torch.load(input_model, map_location=device))
    model.eval()

    # Dummy input for export
    dummy_input = torch.randn(1, input_dim, device=device)

    output_model.parent.mkdir(parents=True, exist_ok=True)

    if format.lower() == "onnx":
        torch.onnx.export(
            model,
            dummy_input,
            output_model,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
            opset_version=11,
        )
        log.info("Model exported to ONNX", path=str(output_model))
    else:
        raise ValueError(f"Unsupported export format: {format}")


if __name__ == "__main__":
    app()

