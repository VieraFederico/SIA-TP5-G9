from dataclasses import dataclass, field
from pathlib import Path
import json


@dataclass
class ExperimentConfig:
    # Identificación
    name: str
    seed: int

    # Datos
    data_path: str
    target_column: str

    preprocessing: str                        # "normalize" | "standardize" | "one_hot"
    split_train: float
    split_val: float
    split_test: float

    # Red
    activation: str                           # "step" | "identity" | "tanh" | "logistic" | "relu"
    beta: float                               # parámetro β para tanh y logistic
    architecture: list[int]                   # neuronas por capa, ej: [784, 64, 10]

    # Entrenamiento
    cost_function: str                        # "mse" | "binary_cross_entropy" | "categorical_cross_entropy"
    optimizer: str                            # "gradient_descent" | "momentum" | "adam"
    eta: float                                # tasa de aprendizaje η
    training_mode: str                        # "online" | "batch" | "minibatch"
    epochs: int                               # máximo de épocas
    epsilon: float                            # umbral de convergencia E < ε

    # Parámetros opcionales de optimizadores
    momentum_beta: float = 0.9
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    batch_size: int = 32

    # Métricas a evaluar al final
    metrics: list[str] = field(default_factory=lambda: ["accuracy"])

    # Opcional lista de columnas a ignorar en el dataset (ej: ID, timestamp)
    columns_to_ignore: list[str] = field(default_factory=list)



def load_config(path: Path) -> ExperimentConfig:
    with open(path) as f:
        data = json.load(f)
    return ExperimentConfig(**data)


def save_config(cfg: ExperimentConfig, path: Path) -> None:
    import dataclasses
    with open(path, "w") as f:
        json.dump(dataclasses.asdict(cfg), f, indent=2)
