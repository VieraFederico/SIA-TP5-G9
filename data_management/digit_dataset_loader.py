import ast

import numpy as np
import pandas as pd


def load_dataset(path: str) -> pd.DataFrame:
    """Load a digits CSV and deserialise the image column to numpy arrays."""
    df = pd.read_csv(path)
    df["image"] = df["image"].apply(
        lambda s: np.array(ast.literal_eval(s), dtype=np.float32)
    )
    return df


def get_image(row: pd.Series, size: tuple[int, int] = (28, 28)) -> np.ndarray:
    """Reshape the flat image vector back to a 2-D array."""
    return row["image"].reshape(size)


if __name__ == "__main__":
    # Import local: rompe el ciclo graphs.image -> este módulo (get_image).
    from graphs.image import plot_sample

    df = load_dataset("digits_test.csv")
    plot_sample(df.iloc[0])
