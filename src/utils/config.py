from dataclasses import dataclass
from pathlib import Path
import json


@dataclass
class Config:
    """Única fuente de verdad de los hiperparámetros del TP (lee config.json).

    Sólo perillas que el pipeline realmente usa (auditoría §4-I): no hay `eta`
    contradictorio (hay un único `learning_rate`) ni metadata muerta. El CLI
    puede overridear cualquiera de estos por flag.
    """
    learning_rate: float                      # lr canónico de AE / VAE
    dae_learning_rate: float                  # lr canónico del DAE (ganador estudio hiperparams-DAE)
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


# config.json vive en la raíz del repo (3 niveles arriba de src/utils/).
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config.json"


def load_config(path: Path | str = DEFAULT_CONFIG_PATH) -> Config:
    """Lee config.json (o el path dado) y devuelve la Config tipada."""
    with open(path) as f:
        data = json.load(f)
    return Config(**data)


def save_config(cfg: Config, path: Path | str = DEFAULT_CONFIG_PATH) -> None:
    import dataclasses
    with open(path, "w") as f:
        json.dump(dataclasses.asdict(cfg), f, indent=2)


# Singleton cargado una vez al importar: ÚNICA fuente de los hiperparámetros del TP.
# Los módulos leen estas constantes desde acá (config), no rebotando por experiment.
# El CLI puede overridearlas por flag.
CFG = load_config()

LEARNING_RATE = CFG.learning_rate
DAE_LEARNING_RATE = CFG.dae_learning_rate    # lr del DAE (≠ AE/VAE); ganador del estudio hiperparams-DAE
EPOCHS = CFG.epochs
BATCH_SIZE = CFG.batch_size          # sólo afecta el modo "minibatch"
EPSILON = CFG.epsilon                # umbral de corte por convergencia (no es el ε de Adam)
TRAINING_MODE = CFG.training_mode
SALT_P = CFG.salt_p
KL_WEIGHT = CFG.kl_weight            # β del VAE
ADAM_BETA1 = CFG.adam_beta1
ADAM_BETA2 = CFG.adam_beta2
SEED = CFG.seed                      # seed por defecto; --seed la overridea
OUTPUT_ROOT = Path(CFG.output_root)  # raíz de las salidas: output/<modelo>/...
