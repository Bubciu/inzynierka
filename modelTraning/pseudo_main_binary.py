"""
This script loads data, processes it, trains a neural network model, and evaluates its performance.
"""

import numpy as np
import torch
from torch import nn
from sklearn.model_selection import train_test_split

import time

from funcs.csv_to_tensor import csv_to_ndarray, ndarray_to_tensor
import funcs.os_operations as oo
from funcs.helper_functions import accuracy_fn, sample, import_model_state_dict, save_model
from funcs.visualisation_functions import ndarray_to_image, ndarray_to_trajectory

from models import ImageModelBinary


IMPORT_MODEL = False
SAVE_MODEL = True
MODEL_PATH = 'correctness/modelsV2'
MODEL_IMPORT_NAME = ''
MODEL_SAVE_NAME = 'model_brzuszek'
"""
Model hyper-parameters.
"""

NUM_CLASSES = 2
"""
Number of output classes for the model.
"""

LEARNING_RATE = 0.00001
EPOCHS = 2000
BATCH_SIZE = 20
"""
Learning parameters.
"""

N_LANDMARKS = 25
"""
Number of landmarks expected in the input data.
"""

N_SAMPLES = 50
"""
Number of samples to use for each instance.
"""

DATA_PATH = "correctness/CSVs/brzuszek"
"""
Path to the directory containing CSV data files.
"""

FILE_EXTENSION = ".csv"

DATA_FORMAT = "plots"
"""
Format of data that goes to model
possible options:
- unchanged
- plots
- trajectories
"""


if __name__ == "__main__":
    """
    Main script execution.

    This block loads the data, preprocesses it, splits it into training and test sets, initializes the model,
    trains the model, evaluates it, and optionally saves the trained model.
    """

    names = oo.files_in_directory(DATA_PATH, FILE_EXTENSION)
    """
    List of data file names in the specified directory with the specified file extension.
    """

    primitive_data = []
    primitive_labels = []

    for i, name in enumerate(names):
        tmp = csv_to_ndarray(fr'{DATA_PATH}/{name}')

        if isinstance(tmp, int) and tmp == 1:
            print(f'{i}: name: {name}\t <dropped>')
            continue

        print(f'{i}: name: {name}\t shape: {tmp.shape}', end='\t')

        if tmp.shape[1] != N_LANDMARKS or tmp.shape[0] < N_SAMPLES:
            print('<dropped>')
            continue

        print('<accepted>')

        tmp = sample(tmp, N_SAMPLES)

        if DATA_FORMAT == "unchanged":
            # to do
            exit(0)
        elif DATA_FORMAT == "plots":
            tmp = ndarray_to_image(tmp)
        elif DATA_FORMAT == "trajectories":
            tmp = ndarray_to_trajectory(tmp)
        else:
            print("Wrong data format")
            exit(0)

        primitive_data.append(tmp)

        # plot_data(tmp)

        if name[0] == 'n':
            primitive_labels.append(0)
        elif name[0] == 'p':
            primitive_labels.append(1)

    primitive_data_stacked = np.array(primitive_data, dtype=np.float32)
    primitive_labels = np.array(primitive_labels, dtype=np.float32)

    X, y = ndarray_to_tensor(primitive_data_stacked, primitive_labels)
    y = y.type(torch.LongTensor)

    print("X shape:", X.shape)
    print("Y shape:", y.shape)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.85, stratify=y)

    print("train size:", X_train.shape)
    print("test size:", X_test.shape)

    if IMPORT_MODEL:
        rm = import_model_state_dict(ImageModelBinary, NUM_CLASSES, MODEL_PATH, MODEL_IMPORT_NAME)
    else:
        rm = ImageModelBinary()

    rm.to(device)

    # if something wrong, try this: nn.BCELoss()
    loss_fn = nn.BCEWithLogitsLoss()
    """
    Loss function for the model, using cross-entropy loss for multi-class classification.
    """

    optimiser = torch.optim.Adam(params=rm.parameters(), lr=LEARNING_RATE)
    """
    Optimizer for the model, using Adam optimizer with the specified learning rate.
    """

    X_train, X_test, y_train, y_test = X_train.to(device), X_test.to(device), y_train.to(device), y_test.to(device)

    t1 = time.time()

    for epoch in range(1, EPOCHS + 1):
        """
        Model training.
        """
        rm.train()
        batch = np.random.choice(X_train.shape[0], BATCH_SIZE, False)
        X_batched = X_train[batch]
        y_logits = rm(X_batched)

        y_logits = y_logits.squeeze()
        y_pred = torch.round(torch.sigmoid(y_logits))

        loss_train = loss_fn(y_logits, y_train[batch].float())
        acc_train = accuracy_fn(y_train[batch], y_pred)

        optimiser.zero_grad()

        loss_train.backward()

        optimiser.step()

        rm.eval()
        with torch.inference_mode():
            """
            Model evaluation.
            """
            y_test_logits = rm(X_test).squeeze()
            y_test_pred = torch.round(torch.sigmoid(y_test_logits))

            loss_test = loss_fn(y_test_logits, y_test.float())
            acc_test = accuracy_fn(y_test, y_test_pred)
            if epoch % 100 == 0:
                print(f'Epoch: {epoch} | Train loss: {loss_train:.5f} | Train acc: {acc_train:.2f}% ||'
                      f'Test loss: {loss_test:.5f} | Test acc: {acc_test:.2f}%')

    t2 = time.time()

    print(f'{round(t2 - t1, 3)}[s]')

    if SAVE_MODEL:
        save_model(rm.cpu(), MODEL_PATH, MODEL_SAVE_NAME)
        """
        Save the trained model to the specified path with the specified filename.
        """
