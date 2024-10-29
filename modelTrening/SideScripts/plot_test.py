import matplotlib.pyplot as plt
import numpy as np
from modelTrening.csv_to_tensor import csv_to_ndarray
from modelTrening.helper_functions import sample


def plot_data(data: np.ndarray) -> None:
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))

    axs[0, 0].imshow(data[:, :, 0])
    axs[0, 0].set_title("x_pos")
    axs[0, 1].imshow(data[:, :, 1])
    axs[0, 1].set_title("y_pos")
    axs[1, 0].imshow(data[:, :, 0] * data[:, :, 1])
    axs[1, 0].set_title("x_pos * y_pos")
    axs[1, 1].axis('off')
    plt.show()


if __name__ == '__main__':
    csv = r"..\CSVs\przysiad7.csv"

    arr = csv_to_ndarray(csv)
    arr = sample(arr, 40)
    new_arr = np.c_[arr[:, :, 0], arr[:, :, 1]]

    plt.figure()
    plt.imshow(arr[:, :, 0])
    plt.show(block=False)

    plt.figure()
    plt.imshow(arr[:, :, 1])
    plt.show(block=False)

    plt.figure()
    plt.imshow(new_arr)
    plt.show()
