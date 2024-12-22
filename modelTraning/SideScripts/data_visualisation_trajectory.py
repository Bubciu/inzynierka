from funcs.os_operations import files_in_directory
from funcs.csv_to_tensor import csv_to_ndarray
from funcs.helper_functions import sample
from funcs.visualisation_functions import visualisation_trajectory

PLOT_SAVE_PATH = "Visualisations/Trajectories"

if __name__ == "__main__":

    names = files_in_directory("Visualisations/CSVs", ".csv")

    for i, name in enumerate(names):
        print(f'{name}')
        tmp = csv_to_ndarray(fr'Visualisations\CSVs\{name}')
        print(f'{i}: name: {name}\t shape: {tmp.shape}', end='\t')

        if tmp.shape[1] != 25 or tmp.shape[0] < 50:
            print('<dropped>')
            continue

        print('<accepted>')

        tmp = sample(tmp, 50)

        visualisation_trajectory(tmp, name.rsplit('.', 1)[0] + ".png", PLOT_SAVE_PATH)
        