# import matplotlib.pyplot as plt
# import numpy as np
from ..os_operations import files_in_directory
from ..csv_to_tensor import csv_to_ndarray
from ..helper_functions import sample
from ..visualisation_functions import visualisation_plot


PLOT_SAVE_PATH = r"..\Visualisations\Plots"

if __name__ == "__main__":

    names = files_in_directory("..\CSVs", ".csv")

    for i, name in enumerate(names):
        print(f'{name}')
        tmp = csv_to_ndarray(fr'..\CSVs\{name}')
        print(f'{i}: name: {name}\t shape: {tmp.shape}', end='\t')

        if tmp.shape[1] != 25:
            print('<dropped>')
            continue

        print('<accepted>')

        tmp = sample(tmp, 40)

        visualisation_plot(tmp, name, PLOT_SAVE_PATH)

