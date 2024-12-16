import pandas as pd
import torch
from pathlib import Path
import numpy as np
import cv2
import time
import mediapipe as mp
import os
import matplotlib.pyplot as plt
import math


def round_list(in_list):
    """
    Rounds values in provided list.
    :param in_list: list of float values to be rounded
    :return: new list with rounded values
    """
    if in_list != in_list:
        in_list = [float('nan'), float('nan')]
    return [round(x, 3) for x in in_list]


def video_to_file(video_path: str, csv_path: str, flip: bool, show_video: bool) -> None:
    """
    Function converts specified video files into landmarks using mediapipe, and later saves as a csv file in a specified location.
    :param video_path: Path of a video file - input file
    :param csv_path: Path of a csv file - output file
    :param flip: Specifies whether to mirror the video or not
    :param show_video: Specifies whether to show the video during processing
    :return: None
    """
    mp_drawing = mp.solutions.drawing_utils
    mp_holistic = mp.solutions.holistic
    alldata = []
    fps_time = 0

    pose_silhouette = ['NOSE', 'LEFT_EYE_INNER', 'LEFT_EYE', 'LEFT_EYE_OUTER', 'RIGHT_EYE_INNER', 'RIGHT_EYE',
                       'RIGHT_EYE_OUTER', 'LEFT_EAR', 'RIGHT_EAR', 'MOUTH_LEFT', 'MOUTH_RIGHT', 'LEFT_SHOULDER',
                       'RIGHT_SHOULDER', 'LEFT_ELBOW', 'RIGHT_ELBOW', 'LEFT_WRIST', 'RIGHT_WRIST', 'LEFT_PINKY',
                       'RIGHT_PINKY', 'LEFT_INDEX', 'RIGHT_INDEX', 'LEFT_THUMB', 'RIGHT_THUMB', 'LEFT_HIP', 'RIGHT_HIP',
                       'LEFT_KNEE', 'RIGHT_KNEE', 'LEFT_ANKLE', 'RIGHT_ANKLE', 'LEFT_HEEL', 'RIGHT_HEEL',
                       'LEFT_FOOT_INDEX', 'RIGHT_FOOT_INDEX']

    exceptions_silhouette = ['LEFT_EYE_INNER', 'LEFT_EYE_OUTER', 'RIGHT_EYE_INNER', 'RIGHT_EYE_OUTER', 'LEFT_EAR',
                             'RIGHT_EAR', 'MOUTH_LEFT', 'MOUTH_RIGHT']

    cap = cv2.VideoCapture(video_path)

    with mp_holistic.Holistic(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                # print(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                break

            if flip:
                image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
            else:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            image.flags.writeable = False
            results = holistic.process(image)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            image_og = np.copy(image)
            image = np.zeros(image.shape)
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)

            data_complete = {}
            if results.pose_landmarks:
                for i, ps in enumerate(pose_silhouette):
                    if ps in exceptions_silhouette:
                        continue

                    results.pose_landmarks.landmark[i].x = results.pose_landmarks.landmark[i].x * 720
                    results.pose_landmarks.landmark[i].y = results.pose_landmarks.landmark[i].y * 1280
                    data_complete.update({ps: [
                        results.pose_landmarks.landmark[i].x, results.pose_landmarks.landmark[i].y]})
            else:
                for i, ps in enumerate(pose_silhouette):
                    data_complete.update({ps: [np.nan, np.nan]})

            alldata.append(data_complete)

            if show_video:
                cv2.putText(image, "FPS: %f" % (1.0 / (time.time() - fps_time)), (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0, 255, 0), 2, )
                cv2.imshow('MediaPipe Holistic', image)
                cv2.imshow('Original Image', image_og)
                fps_time = time.time()
            if cv2.waitKey(5) & 0xFF == 27:
                break

    cap.release()

    df = pd.DataFrame(alldata)
    # df.drop(columns=exceptions_silhouette, inplace=True)
    # for i in df:
    #     fill_missing_values_mean(df[i])

    df = df.applymap(lambda x: round_list(x))
    df.to_csv(csv_path, sep=';', index=False, header=False)


def fill_missing_values_mean(column: pd.Series) -> None:
    """
    Fills missing values into column based on surrounding values
    :param column: values to be filled if missing
    :return: None
    """

    column_array = [item for item in column]
    nan_indices = np.where(np.isnan(np.array(column_array)).all(axis=1))[0]

    while len(nan_indices) > 0:
        prev_idx = None
        count = 0
        min_idx = None
        middle_idx_tab = []
        for idx in nan_indices:
            if prev_idx is None:
                count += 1
                min_idx = idx
                prev_idx = idx
                continue

            if prev_idx == idx - 1:
                count += 1
                prev_idx = idx
            else:
                middle_idx_tab.append(min_idx + int(count / 2))
                nan_indices = np.delete(nan_indices, np.where(nan_indices == min_idx + int(count / 2)))
                min_idx = idx
                prev_idx = idx
                count = 1

        middle_idx_tab.append(min_idx + int(count / 2))
        nan_indices = np.delete(nan_indices, np.where(nan_indices == min_idx + int(count / 2)))

        for idx in middle_idx_tab:
            left_idx_x = idx - 1
            right_idx_x = idx + 1
            left_idx_y = idx - 1
            right_idx_y = idx + 1

            if right_idx_x > len(column_array) - 1:
                if not np.isnan(column_array[idx - 1][0]) and not np.isnan(column_array[idx - 2][0]):
                    column_array[idx][0] = column_array[idx - 1][0] + (
                                column_array[idx - 1][0] - column_array[idx - 2][0])
                    column_array[idx][1] = column_array[idx - 1][1] + (
                                column_array[idx - 1][1] - column_array[idx - 2][1])
                else:
                    nan_indices = np.append(nan_indices, idx)
                    for i in range(idx - 1):
                        if not np.isnan(column_array[idx - i - 1][0]) and not np.isnan(column_array[idx - i - 2][0]):
                            column_array[idx - i][0] = column_array[idx - i - 1][0] + (
                                        column_array[idx - i - 1][0] - column_array[idx - i - 2][0])
                            column_array[idx - i][1] = column_array[idx - i - 1][1] + (
                                        column_array[idx - i - 1][1] - column_array[idx - i - 2][1])
                            break
                continue

            if left_idx_x < 0:
                if not np.isnan(column_array[idx + 1][0]) and not np.isnan(column_array[idx + 2][0]):
                    column_array[idx][0] = column_array[idx + 1][0] - (
                                column_array[idx + 2][0] - column_array[idx + 1][0])
                    column_array[idx][1] = column_array[idx + 1][1] - (
                                column_array[idx + 2][1] - column_array[idx + 1][1])
                else:
                    nan_indices = np.append(nan_indices, idx)
                    for i in range(len(column_array) - idx - 1):
                        if not np.isnan(column_array[idx + i + 1][0]) and not np.isnan(column_array[idx + i + 2][0]):
                            column_array[idx + i][0] = column_array[idx + i + 1][0] - (
                                        column_array[idx + i + 2][0] - column_array[idx + i + 1][0])
                            column_array[idx + i][1] = column_array[idx + i + 1][1] - (
                                        column_array[idx + i + 2][1] - column_array[idx + i + 1][1])
                            break
                continue

            while np.isnan(column_array[left_idx_x][0]) and left_idx_x > 0:
                left_idx_x -= 1

            while np.isnan(column_array[right_idx_x][0]) and right_idx_x < len(column_array) - 1:
                right_idx_x += 1

            while np.isnan(column_array[left_idx_y][1]) and left_idx_y > 0:
                left_idx_y -= 1

            while np.isnan(column_array[right_idx_y][1]) and right_idx_y < len(column_array) - 1:
                right_idx_y += 1

            x_left = column_array[left_idx_x][0]
            x_right = column_array[right_idx_x][0]
            y_left = column_array[left_idx_y][1]
            y_right = column_array[right_idx_y][1]

            column_array[idx][0] = (x_left + x_right) / 2
            column_array[idx][1] = (y_left + y_right) / 2

    column = pd.DataFrame([column_array])


def accuracy_fn(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    """
    Calculates accuracy of classification/regression in percentages
    :param y_true: tensor of target values
    :param y_pred: tensor of predicted values
    :return: Percentage accuracy of the prediction
    """
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct/len(y_pred)) * 100
    return acc


def import_model_state_dict(model_class: torch.nn, num_class: int, path: str, model_name: str) -> torch.nn:
    """
    Creates model with loaded state dictionary
    :param model_class: model class name
    :param num_class: number of model decision classes
    :param path: path to the model
    :param model_name: model filename
    :return: ready model
    """
    model_path = Path(path)
    model_load_name = f'{model_name}.pth'
    full_path = model_path / model_load_name
    model = model_class(num_class)
    model.load_state_dict(torch.load(full_path))
    return model


def save_model(model: torch.nn, path: str, model_name: str) -> None:
    """
    Save models state dict
    :param model: taught model (derived from torch.nn)
    :param path: save path
    :param model_name: filename
    :return: None
    """
    model_path = Path(path)
    model_path.mkdir(parents=True, exist_ok=True)

    model_name = f'{model_name}.pth'
    full_path = model_path / model_name

    print(f'Saving model to: {full_path}')
    torch.save(obj=model.state_dict(), f=full_path)


def extract_landmarks_from_video() -> pd.DataFrame:
    """
    :)
    :return: We thought that it would do something, but apparently we forgot. So, yeh. Just a pass statement.
    """
    pass


def sample(data: np.ndarray, number_of_samples: int) -> np.ndarray:
    """
    Samples provided ndarray.
    :param data: ndarray with landmark coordinates
    :param number_of_samples: number of samples to take from data
    :return: sampled ndarray
    """
    return data[np.sort(np.random.choice(data.shape[0], number_of_samples, False))]


def delete_files_in_directory(directory_path):
    """
    Removes all files from the specified directory.
    :param directory_path: path from which files are to be annihilated (enters boss music)
    :return: None (there's only dust and smoke left in the directory)
    """

    try:
        files = os.listdir(directory_path)
        for file in files:
            file_path = os.path.join(directory_path, file)
            if os.path.isfile(file_path):
                os.remove(file_path)
        print("All files deleted successfully.")
    except OSError:
        print("Error occurred while deleting files.")
