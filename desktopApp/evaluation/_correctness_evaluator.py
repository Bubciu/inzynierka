import numpy as np
import pandas as pd
import torch
import warnings

from os.path import abspath
from pathlib import Path
from ._dependencies import CorectnessEvaluationModel, sample, round_list, ndarray_to_image
from math import ceil

warnings.simplefilter(action='ignore', category=FutureWarning)


class CorrectnessEvaluator:
    """
    Class for evaluating correctness of an exercise data using a pre-trained model.
    """

    def __init__(self, exercise_idx):
        """
        Initialises the CorrectnessEvaluator with a pre-trained model.
        """

        self.__model = CorectnessEvaluationModel()

        # load model, based on param
        if exercise_idx == 1:
            self.__model.load_state_dict(torch.load(
                abspath(fr"{Path(__file__).resolve().parent}\_model\model_pajac.pth")))
        elif exercise_idx == 2:
            self.__model.load_state_dict(torch.load(
                abspath(fr"{Path(__file__).resolve().parent}\_model\model_przysiadBok.pth")))
        elif exercise_idx == 3:
            self.__model.load_state_dict(torch.load(
                abspath(fr"{Path(__file__).resolve().parent}\_model\model_przysiad.pth")))
        elif exercise_idx == 4:
            self.__model.load_state_dict(torch.load(
                abspath(fr"{Path(__file__).resolve().parent}\_model\model_brzuszek.pth")))
        elif exercise_idx == 5:
            self.__model.load_state_dict(torch.load(
                abspath(fr"{Path(__file__).resolve().parent}\_model\model_sklonBok.pth")))
        elif exercise_idx == 6:
            self.__model.load_state_dict(torch.load(
                abspath(fr"{Path(__file__).resolve().parent}\_model\model_sklon.pth")))
        else:
            print("błąd ładowania modelu")
            return

        self.__device = "cuda" if torch.cuda.is_available() else "cpu"
        self.__model.to(self.__device)
        self.__model.eval()

        dummy_input = torch.randn(300, 300, 3).to(self.__device)
        self.__model = torch.jit.trace(self.__model, dummy_input)

        self.__model_frames = 50

    def evaluate_data(self, data):
        """
        Evaluates the provided exercise data and returns the predicted class.

        :param data: List of landmarks representing the exercise data.
        :return: 1 if exercise sent was done correctly and 0 if done incorrectly.
        """

        if len(data) < self.__model_frames:
            data_extended = self.__extend_data(data)
            df = pd.DataFrame(data_extended)
        else:
            df = pd.DataFrame(data)

        df = df.applymap(lambda x: round_list(x))

        data_array = np.array([row.tolist() for _, row in df.iterrows()])
        image = ndarray_to_image(sample(data_array, self.__model_frames))
        tensor = torch.from_numpy(image).type(torch.float32).to(self.__device)

        with torch.inference_mode():
            logit = self.__model(tensor)
            prob = torch.sigmoid(logit)
            predicted_class = (prob >= 0.5).long().item()

        return predicted_class

    def __extend_data(self, data):
        """
        Extends the data to match the required number of frames for the model.

        :param data: List of landmarks representing the exercise data.
        :return: Extended data.
        """

        multiplier = ceil(self.__model_frames / len(data))
        data_extended = [item for item in data for _ in range(multiplier)]
        return data_extended
