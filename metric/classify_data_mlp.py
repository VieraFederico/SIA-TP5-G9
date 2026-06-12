import numpy as np
from src.activation.activation import Array


def classify_data_mlp (zeta: Array, output: Array)->Array:

    arr_output_size = output.shape[0]
    arr_zeta_size = zeta.shape[0]
    if arr_output_size != arr_zeta_size:
        raise ValueError("zeta and output have different shapes")

    confusion_matrix = np.zeros((10, 10), dtype=int)
    predictions = np.argmax(output, axis=1)

    for index in range(arr_output_size):
        confusion_matrix[zeta[index]][predictions[index]] += 1

    return confusion_matrix
