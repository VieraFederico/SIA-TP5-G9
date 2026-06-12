from typing import List

import numpy as np
from src.data_management.dataset import Dataset


def train_val_test_split(
    dataset: Dataset,
    train: float,
    val: float,
    test: float,
    seed: int,
) -> tuple[Dataset, Dataset, Dataset]:
    """Divide dataset en train/val/test con proporciones dadas."""
    return dataset.split(train=train, val=val, test=test, seed=seed)

def k_fold_split(dataset: Dataset, k: int, seed: int) -> list[tuple[Dataset, Dataset]]:
    """Genera k particiones (train, val) para validación cruzada."""
    n = len(dataset.X)
    rng = np.random.default_rng(seed)
    indices = rng.permutation(n)
    folds = np.array_split(indices, k)
    splits = []
    for i in range(k):
        val_idx = folds[i]
        train_idx = np.concatenate([folds[j] for j in range(k) if j != i])
        train_ds = Dataset(dataset.X[train_idx], dataset.zeta[train_idx])
        val_ds = Dataset(dataset.X[val_idx], dataset.zeta[val_idx])
        splits.append((train_ds, val_ds))
    return splits



def k_fold_split_v2(dataset: Dataset, k: int, seed: int) -> list[Dataset]:
    """Takes a Dataset, and randomly splits it into k folds."""
    n = len(dataset.X)
    rng = np.random.default_rng(seed)
    indices = rng.permutation(n)
    folds = np.array_split(indices, k)
    fold_datasets = []
    for fold_idx in folds:
        fold_ds = Dataset(dataset.X[fold_idx], dataset.zeta[fold_idx])
        fold_datasets.append(fold_ds)

    return fold_datasets
