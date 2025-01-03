import numpy as np
import pandas as pd
import torch
import warnings

from os.path import abspath
from pathlib import Path
from ._dependencies import CorectnessEvaluationModel, CorectnessEvaluationImageModel, sample, round_list, ndarray_to_plot, ndarray_to_trajectory
from math import ceil

warnings.simplefilter(action='ignore', category=FutureWarning)


class CorrectnessEvaluator:
    """
    Class for evaluating correctness of an exercise data using a pre-trained model.
    """

    def __init__(self, exercise_idx, data_format):
        """
        Initialises the CorrectnessEvaluator with a pre-trained model.
        """

        if data_format == 'unchanged':
            self.__model = CorectnessEvaluationModel()
        elif data_format == 'plot' or data_format == 'trajectory':
            self.__model = CorectnessEvaluationImageModel()

        if exercise_idx == 1:
            self.__model.load_state_dict(torch.load(
                abspath(fr"{Path(__file__).resolve().parent}\_model\{data_format}\model_pajac.pth")))
        elif exercise_idx == 2:
            self.__model.load_state_dict(torch.load(
                abspath(fr"{Path(__file__).resolve().parent}\_model\{data_format}\model_przysiadBok.pth")))
        elif exercise_idx == 3:
            self.__model.load_state_dict(torch.load(
                abspath(fr"{Path(__file__).resolve().parent}\_model\{data_format}\model_przysiad.pth")))
        elif exercise_idx == 4:
            self.__model.load_state_dict(torch.load(
                abspath(fr"{Path(__file__).resolve().parent}\_model\{data_format}\model_brzuszek.pth")))
        elif exercise_idx == 5:
            self.__model.load_state_dict(torch.load(
                abspath(fr"{Path(__file__).resolve().parent}\_model\{data_format}\model_sklonBok.pth")))
        elif exercise_idx == 6:
            self.__model.load_state_dict(torch.load(
                abspath(fr"{Path(__file__).resolve().parent}\_model\{data_format}\model_sklon.pth")))
        else:
            print("błąd ładowania modelu")
            return

        self.__device = "cuda" if torch.cuda.is_available() else "cpu"
        self.__model.to(self.__device)
        self.__model.eval()

        if data_format == 'unchanged':
            dummy_input = torch.randn(1, 50, 25, 2).to(self.__device)
        elif data_format == 'plot' or data_format == 'trajectory':
            dummy_input = torch.randn(300, 300, 3).to(self.__device)
        self.__model = torch.jit.trace(self.__model, dummy_input)

        self.__model_frames = 50
        self.__data_format = data_format

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
        sampled = sample(data_array, self.__model_frames)
        
        if self.__data_format == 'unchanged':
            data = sampled
            data = np.expand_dims(data, axis=0)
        elif self.__data_format == 'plot':
            data = ndarray_to_plot(sampled)
        elif self.__data_format == 'trajectory':
            data = ndarray_to_trajectory(sampled)
        else:
            print("wrong data format")
            return 1

        tensor = torch.from_numpy(data).type(torch.float32).to(self.__device)

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
