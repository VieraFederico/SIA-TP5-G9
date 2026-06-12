from dataclasses import dataclass
from src.activation.activation import Array
import numpy as np


@dataclass
class Dataset:
    """Contiene X (entradas) y zeta (salidas esperadas ζ)."""

    X: Array
    zeta: Array

    def split(self, train=0.7, val=0.0, test=0.3, seed=42) -> tuple["Dataset", "Dataset", "Dataset", list[np.ndarray]]:
        assert abs(train + val + test - 1.0) < 1e-6, "train+val+test must sum to 1"

        n = len(self.X)

        rng = np.random.default_rng(seed)
        indices = rng.permutation(n)

        train_end = int(train * n)
        val_end = train_end + int(val * n)

        train_idx = indices[:train_end]
        val_idx = indices[train_end:val_end]
        test_idx = indices[val_end:]

        train_ds = Dataset(self.X[train_idx], self.zeta[train_idx])
        val_ds = Dataset(self.X[val_idx], self.zeta[val_idx])
        test_ds = Dataset(self.X[test_idx], self.zeta[test_idx])

        return train_ds, val_ds, test_ds, [train_idx, val_idx, test_idx]

    def copy(self):
        return Dataset(self.X.copy(), self.zeta.copy())