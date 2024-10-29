"""
This script loads data, processes it, trains a neural network model, and evaluates its performance.
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from sklearn.model_selection import train_test_split

import time

from csv_to_tensor import csv_to_ndarray, ndarray_to_tensor
import os_operations as oo
from helper_functions import accuracy_fn, sample, import_model_state_dict, save_model
from visualisation_functions import ndarray_to_image

from test_model import TestModel


IMPORT_MODEL = False
SAVE_MODEL = True
MODEL_PATH = 'Models'
MODEL_IMPORT_NAME = 'model_la_studentos_v2'
MODEL_SAVE_NAME = 'model_v5'
"""
Model hyper-parameters.
"""

NUM_CLASSES = 7
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

DATA_PATH = "CSVs"
"""
Path to the directory containing CSV data files.
"""

FILE_EXTENSION = ".csv"

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
        # sep = '/' if oo.IS_WINDOWS else '\\'
        tmp = csv_to_ndarray(fr'{DATA_PATH}/{name}')
        print(f'{i}: name: {name}\t shape: {tmp.shape}', end='\t')

        if tmp.shape[1] != N_LANDMARKS or tmp.shape[0] < N_SAMPLES:
            print('<dropped>')
            continue

        print('<accepted>')

        tmp = sample(tmp, N_SAMPLES)
        tmp = ndarray_to_image(tmp)
        primitive_data.append(tmp)

        #plot_data(tmp)

        if 'nic' in name:
            primitive_labels.append(0)
        elif 'pajac' in name:
            primitive_labels.append(1)
        elif 'przysiadBok' in name:
            primitive_labels.append(2)
        elif "przysiad" in name:
            primitive_labels.append(3)
        elif "brzuszek" in name:
            primitive_labels.append(4)
        elif "sklonBok" in name:
            primitive_labels.append(5)
        elif "sklon" in name:
            primitive_labels.append(6)

    primitive_data_stacked = np.array(primitive_data, dtype=np.float32)
    primitive_labels = np.array(primitive_labels, dtype=np.float32)

    X, y = ndarray_to_tensor(primitive_data_stacked, primitive_labels)
    y = y.type(torch.LongTensor)

    print(X.shape)
    print(y.shape)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.9, stratify=y)

    print("train size:", X_train.shape)
    print("test size:", X_test.shape)

    if IMPORT_MODEL:
        rm = import_model_state_dict(TestModel, NUM_CLASSES, MODEL_PATH, MODEL_IMPORT_NAME)
    else:
        rm = TestModel(NUM_CLASSES)

    rm.to(device)

    loss_fn = nn.CrossEntropyLoss()
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
        # y_pred = torch.round(torch.sigmoid(y_logits))          # bianry
        y_pred = torch.softmax(y_logits, dim=1).argmax(dim=1)    # multiclass

        loss_train = loss_fn(y_logits, y_train[batch])
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
            y_test_pred = torch.softmax(y_test_logits, dim=1).argmax(dim=1)

            loss_test = loss_fn(y_test_logits, y_test)
            acc_test = accuracy_fn(y_test, y_test_pred)
            if epoch % 50 == 0:
                print(f'Epoch: {epoch} | Train loss: {loss_train:.5f} | Train acc: {acc_train:.2f}% ||'
                      f'Test loss: {loss_test:.5f} | Test acc: {acc_test:.2f}%')

    t2 = time.time()

    print(f'{round(t2 - t1, 3)}[s]')

    if SAVE_MODEL:
        save_model(rm.cpu(), MODEL_PATH, MODEL_SAVE_NAME)
        """
        Save the trained model to the specified path with the specified filename.
        """
