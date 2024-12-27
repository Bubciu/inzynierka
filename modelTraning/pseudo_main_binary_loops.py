import numpy as np
import torch
from torch import nn
from sklearn.model_selection import train_test_split
import time
from funcs.csv_to_tensor import csv_to_ndarray, ndarray_to_tensor
import funcs.os_operations as oo
from funcs.helper_functions import accuracy_fn, sample, import_model_state_dict, save_model
from funcs.visualisation_functions import ndarray_to_image, ndarray_to_trajectory
from models import ModelBinary, ImageModelBinary

EXERCISES = ['brzuszek', 'pajac', 'przysiad', 'przysiadBok', 'sklon', 'sklonBok']

NUM_CLASSES = 2
LEARNING_RATE = 0.00001
EPOCHS = 2000
BATCH_SIZE = 20
N_LANDMARKS = 25
N_SAMPLES = 50
FILE_EXTENSION = ".csv"
IMPORT_MODEL = False
SAVE_MODEL = True

DATA_FORMAT = 'trajectories'
# unchanged plots trajectories
MODEL_PATH = f'correctness/modelsTrajectoryV2'

for exercise in EXERCISES:
    DATA_PATH = f'correctness/CSVs/{exercise}'
    MODEL_SAVE_NAME = f'model_{exercise}'
    
    print(f'--- Training for {exercise} with {DATA_FORMAT} format ---')

    names = oo.files_in_directory(DATA_PATH, FILE_EXTENSION)
    primitive_data = []
    primitive_labels = []

    for i, name in enumerate(names):
        tmp = csv_to_ndarray(fr'{DATA_PATH}/{name}')

        if isinstance(tmp, int) and tmp == 1:
            continue

        if tmp.shape[1] != N_LANDMARKS or tmp.shape[0] < N_SAMPLES:
            continue

        tmp = sample(tmp, N_SAMPLES)

        if DATA_FORMAT == "plots":
            tmp = ndarray_to_image(tmp)
        elif DATA_FORMAT == "trajectories":
            tmp = ndarray_to_trajectory(tmp)

        primitive_data.append(tmp)
        primitive_labels.append(0 if name[0] == 'n' else 1)

    primitive_data_stacked = np.array(primitive_data, dtype=np.float32)
    primitive_labels = np.array(primitive_labels, dtype=np.float32)

    X, y = ndarray_to_tensor(primitive_data_stacked, primitive_labels)
    y = y.type(torch.LongTensor)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.85, stratify=y)

    print("train size:", X_train.shape)
    print("test size:", X_test.shape)

    if IMPORT_MODEL:
        model_class = ImageModelBinary if DATA_FORMAT in ['plots', 'trajectories'] else ModelBinary
        rm = import_model_state_dict(model_class, MODEL_PATH, MODEL_SAVE_NAME)
    else:
        rm = ImageModelBinary() if DATA_FORMAT in ['plots', 'trajectories'] else ModelBinary()
    
    rm.to(device)

    loss_fn = nn.BCEWithLogitsLoss()
    optimiser = torch.optim.Adam(params=rm.parameters(), lr=LEARNING_RATE)

    X_train, X_test, y_train, y_test = X_train.to(device), X_test.to(device), y_train.to(device), y_test.to(device)

    t1 = time.time()

    for epoch in range(1, EPOCHS + 1):
        rm.train()
        batch = np.random.choice(X_train.shape[0], BATCH_SIZE, False)
        X_batched = X_train[batch]
        y_logits = rm(X_batched).squeeze()
        y_pred = torch.round(torch.sigmoid(y_logits))
        loss_train = loss_fn(y_logits, y_train[batch].float())
        acc_train = accuracy_fn(y_train[batch], y_pred)
        optimiser.zero_grad()
        loss_train.backward()
        optimiser.step()

        rm.eval()
        with torch.inference_mode():
            y_test_logits = rm(X_test).squeeze()
            y_test_pred = torch.round(torch.sigmoid(y_test_logits))
            loss_test = loss_fn(y_test_logits, y_test.float())
            acc_test = accuracy_fn(y_test, y_test_pred)

            if epoch % 100 == 0:
                print(f'Epoch: {epoch} | Train loss: {loss_train:.5f} | Train acc: {acc_train:.2f}% || '
                        f'Test loss: {loss_test:.5f} | Test acc: {acc_test:.2f}%')

    t2 = time.time()
    print(f'Training time: {round(t2 - t1, 3)}[s]')

    if SAVE_MODEL:
        save_model(rm.cpu(), MODEL_PATH, MODEL_SAVE_NAME)
        print(f'Model saved as {MODEL_SAVE_NAME} in {MODEL_PATH}')
