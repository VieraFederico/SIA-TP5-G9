from typing import List

from src.data_management.dataset import Dataset
import pandas as pd
import numpy as np


def load_csv(path: str, target_column: str, columns_to_ignore=None) -> Dataset:
    """Lee un CSV y devuelve un Dataset con X y zeta."""
    if columns_to_ignore is None:
        columns_to_ignore = []
    df = pd.read_csv(path)

    if target_column not in df.columns:
        raise ValueError(f"target_column '{target_column}' not found in CSV columns")

    # Drop ignored columns if they exist
    if columns_to_ignore:
        missing = [c for c in columns_to_ignore if c not in df.columns]
        if missing:
            raise ValueError(f"columns_to_ignore not found in CSV columns: {missing}")
        df = df.drop(columns=columns_to_ignore)

    zeta = df[target_column].to_numpy(dtype=np.float64)
    X = df.drop(columns=[target_column]).to_numpy(dtype=np.float64)

    return Dataset(X=X, zeta=zeta)