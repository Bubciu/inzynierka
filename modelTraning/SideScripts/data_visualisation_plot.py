from ..os_operations import files_in_directory
from ..csv_to_tensor import csv_to_ndarray
from ..helper_functions import sample
from ..visualisation_functions import visualisation_plot


PLOT_SAVE_PATH = "modelTraning/Visualisations/Plots"

if __name__ == "__main__":

    names = files_in_directory("modelTraning/Visualisations/CSVs", ".csv")

    for i, name in enumerate(names):
        print(f'{name}')
        tmp = csv_to_ndarray(fr'modelTraning\Visualisations\CSVs\{name}')
        print(f'{i}: name: {name}\t shape: {tmp.shape}', end='\t')

        if tmp.shape[1] != 25 or tmp.shape[0] < 50:
            print('<dropped>')
            continue

        print('<accepted>')

        tmp = sample(tmp, 50)

        visualisation_plot(tmp, name.rsplit('.', 1)[0] + ".png", PLOT_SAVE_PATH)

