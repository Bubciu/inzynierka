import numpy as np
import pandas as pd
import torch
import warnings

from os.path import abspath
from pathlib import Path
from ._dependencies import ExerciseEvaluationModel, ExerciseEvaluationImageModel, sample, round_list, ndarray_to_plot, ndarray_to_trajectory
from math import ceil


warnings.simplefilter(action='ignore', category=FutureWarning)


available_classes = {
    0: "None",
    1: "Jumping Jack",
    2: "Side Leg Squat",
    3: "Squat",
    4: "Standing Sit-up",
    5: "Side Bend",
    6: "Bend"
}
"""
:brief: Dictionary of pairs idx[int]: exercise_name[str] showcasing which exercise_id's can be passed to
        ExerciseEvaluator.evaluate_data method
"""


class ExerciseEvaluator:
    """
    Class for evaluating exercise data using a pre-trained model.
    """

    def __init__(self, data_format):
        """
        Initialises the ExerciseEvaluator with a pre-trained model.
        """
        if data_format == 'unchanged':
            self.__model = ExerciseEvaluationModel(7)
        elif data_format == 'plot' or data_format == 'trajectory':
            self.__model = ExerciseEvaluationImageModel(7)
        self.__model.load_state_dict(torch.load(abspath(fr"{Path(__file__).resolve().parent}\_model\{data_format}\model.pth")))
        self.__device = "cuda" if torch.cuda.is_available() else "cpu"
        print(self.__device)
        self.__model.to(self.__device)
        self.__model.eval()

        if data_format == 'unchanged':
            dummy_input = torch.randn(1, 50, 25, 2).to(self.__device)
        elif data_format == 'plot' or data_format == 'trajectory':
            dummy_input = torch.randn(300, 300, 3).to(self.__device)

        self.__model = torch.jit.trace(self.__model, dummy_input)

        self.__finding_cutoff = 0.8
        self.__model_frames = 50
        self.__data_format = data_format


    def evaluate_data(self, data, expected_class=None):
        """
        Evaluates the provided exercise data and returns the predicted class.

        :param data: List of landmarks representing the exercise data.
        :param expected_class: The expected class index (optional).
        :return: Predicted class index or top 3 classes with probabilities if expected_class is None. -1 on error.
        """

        if expected_class is not None and expected_class not in list(available_classes.keys()):
            return -1

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

        tensor = torch.from_numpy(data).to(self.__device, dtype=torch.float32)

        with torch.inference_mode():
            logit = self.__model(tensor)
            probs = torch.softmax(logit, dim=1)
            assigned_class_idx = probs.argmax(dim=1).item()

            if probs.max() > self.__finding_cutoff and assigned_class_idx != 0:
                if assigned_class_idx == expected_class or expected_class is None:
                    return assigned_class_idx

        return 0

    def __extend_data(self, data):
        """
        Extends the data to match the required number of frames for the model.

        :param data: List of landmarks representing the exercise data.
        :return: Extended data.
        """

        multiplier = ceil(self.__model_frames / len(data))
        data_extended = [item for item in data for _ in range(multiplier)]
        return data_extended
