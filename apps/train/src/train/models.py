from pathlib import Path
from pydantic import BaseModel, Field


class TrainParams(BaseModel):
    backbone: str = Field("resnet18", description="Model backbone to use")
    epochs: int = Field(10, description="Number of training epochs")
    batch_size: int = Field(32, description="Batch size")
    lr: float = Field(0.001, description="Learning rate")
    dropout: float = Field(0.2, description="Dropout rate")
