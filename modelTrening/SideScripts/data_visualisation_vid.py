import cv2
import numpy as np
from modelTrening.csv_to_tensor import csv_to_ndarray
from modelTrening.visualisation_functions import visualisation_video
import os

if __name__ == "__main__":

    name = "przysiadBok2"
    landmarks_array = csv_to_ndarray(fr"../CSVs/{name}.csv")

    fps = 15
    width, height = 1280, 720

    # output_video_path = fr"../Visualisations/Videos"
    output_video_path = os.path.abspath("..")
    output_video_name = f"{name}.avi"

    visualisation_video(landmarks_array, output_video_name, output_video_path, fps, height, width)
