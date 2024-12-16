import torch
import numpy as np
from ast import literal_eval


def csv_to_ndarray(csv_path: str) -> np.array:
    """
    Function takes csv path in string format,
    goes through each line and converts string values to list
    Lastly, it creates numpy array of extracted data
    :param csv_path: path to a file in a string format
    :return: ndarray from csv
    """

    outer = list()

    with open(csv_path, 'r') as csv:
        while True:
            line = csv.readline().split(';')
            # print(line)
            if line == ['']:
                break

            inner = list()
            for val in line:
                try:
                    elem = literal_eval(val)
                except (ValueError, SyntaxError) as e:
                    return 1
                inner.append(elem)
            outer.append(inner)

    return np.array(outer)


def ndarray_to_tensor(X: np.array, y: np.array) -> (torch.Tensor, torch.Tensor):
    """
    Function creates torch.Tensor's from provided arrays
    :param X: data vector/matrix
    :param y: target vector
    :return: tuple of tensors representing X, y respectively
    """
    return torch.from_numpy(X).type(torch.float32), torch.from_numpy(y).type(torch.float32)
