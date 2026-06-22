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



@dataclass
class Config:
    """Única fuente de verdad de los hiperparámetros del TP (lee config.json).

    Sólo perillas que el pipeline realmente usa (auditoría §4-I): no hay `eta`
    contradictorio (hay un único `learning_rate`) ni metadata muerta. El CLI
    puede overridear cualquiera de estos por flag.
    """
    learning_rate: float
    epochs: int
    training_mode: str                        # "online" | "batch" | "minibatch"
    batch_size: int                           # sólo se usa en modo "minibatch"
    epsilon: float                            # umbral de convergencia E < ε
    salt_p: float                             # nivel de ruido Salt & Pepper del DAE
    kl_weight: float                          # β del VAE (vae.KL_WEIGHT)
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    seed: int | None = None                   # default de reproducibilidad; --seed lo overridea
    output_root: str = "output"               # raíz de las salidas generadas


# config.json vive en la raíz del repo, al lado de este archivo.
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "config.json"


def load_config(path: Path | str = DEFAULT_CONFIG_PATH) -> Config:
    """Lee config.json (o el path dado) y devuelve la Config tipada."""
    with open(path) as f:
        data = json.load(f)
    return Config(**data)


def save_config(cfg: Config, path: Path | str = DEFAULT_CONFIG_PATH) -> None:
    import dataclasses
    with open(path, "w") as f:
        json.dump(dataclasses.asdict(cfg), f, indent=2)
