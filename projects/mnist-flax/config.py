from dataclasses import dataclass


@dataclass
class Config:
    batch_size: int = 512
    lr: float = 0.1
    momentum: float = 0.9
    num_epochs: int = 20
