import cv2
import matplotlib.pyplot as plt
import mediapipe as mp
import numpy as np
import pandas as pd
from helper_functions import sample, round_list, delete_files_in_directory
from pathlib import Path
from test_model import TestModel
import torch
from os.path import abspath
from math import ceil
from visualisation_functions import visualisation_video, ndarray_to_image
import warnings


warnings.simplefilter(action='ignore', category=FutureWarning)


class TestApp:
    """
    Prototype class showcasing and testing taught model capabilities.
    """
    def __init__(self, model: torch.nn):
        """
        just an initialiser
        :param model: taught model
        """

        self.__model = model
        self.__device = "cuda" if torch.cuda.is_available() else "cpu"
        self.__model.to(self.__device)
        self.__classes = {
            0: "None",
            1: "Jumping Jack",
            2: "Side Leg Squat",
            3: "Squat",
            4: "Standing Sit-up",
            5: "Side Bend",
            6: "Bend",
        }

        self.__frames_per_exercise = {
            1: [25, 20],    # video 4/4, jest meh, pajacyk jest dalej za krótki
            2: [50, 40],   # video not tested, kamera picus glancus
            3: [40, 30],    # video 4/4, kamera łapie ładnie, nie można za szybko robić
            4: [25, 20],    # video not tested, kamera łapie, ale ćwiczenie za szybkie
            5: [70, 60],   # video 6/4, dobrze wystarczająco, długie ćwiczenie, potrzebuje więcej klatek
            6: [80, 60],   # video 3/3, początek trzeba poczekać, ale później działa ładnie
        }
        # można dodać znak startu po pierwszej decyzji modelu :D

        self.__current_exercise = None
        self.__pose_silhouette = ['NOSE', 'LEFT_EYE_INNER', 'LEFT_EYE', 'LEFT_EYE_OUTER', 'RIGHT_EYE_INNER',
                                  'RIGHT_EYE', 'RIGHT_EYE_OUTER', 'LEFT_EAR', 'RIGHT_EAR', 'MOUTH_LEFT', 'MOUTH_RIGHT',
                                  'LEFT_SHOULDER', 'RIGHT_SHOULDER', 'LEFT_ELBOW', 'RIGHT_ELBOW', 'LEFT_WRIST',
                                  'RIGHT_WRIST', 'LEFT_PINKY', 'RIGHT_PINKY', 'LEFT_INDEX', 'RIGHT_INDEX', 'LEFT_THUMB',
                                  'RIGHT_THUMB', 'LEFT_HIP', 'RIGHT_HIP', 'LEFT_KNEE', 'RIGHT_KNEE', 'LEFT_ANKLE',
                                  'RIGHT_ANKLE', 'LEFT_HEEL', 'RIGHT_HEEL', 'LEFT_FOOT_INDEX', 'RIGHT_FOOT_INDEX']
        self.__exceptions_silhouette = ['LEFT_EYE_INNER', 'LEFT_EYE_OUTER', 'RIGHT_EYE_INNER', 'RIGHT_EYE_OUTER',
                                        'LEFT_EAR', 'RIGHT_EAR', 'MOUTH_LEFT', 'MOUTH_RIGHT']
        self.__frames_per_eval = None
        self.__cut_frames_find = None
        self.__cut_frames_nothing = 10
        self.__finding_cutoff = 0.8
        #best results with: 80, 60, 10, 0.8
        self.__model_frames = 50
        self.__i = 0
        # DELETE THIS LATER!!!!!
        self.__amount = 0;

    def __extract_landmarks(self, landmark_list):
        """
        Extracts landmarks from provided list and normalises them.
        :param landmark_list: list of mediapipe landmarks
        :return: dictionary of landmarks
        """
        frame_landmarks = {}
        if landmark_list.pose_landmarks:
            for i, ps in enumerate(self.__pose_silhouette):
                if ps in self.__exceptions_silhouette:
                    continue

                landmark_list.pose_landmarks.landmark[i].x = landmark_list.pose_landmarks.landmark[i].x * 720
                landmark_list.pose_landmarks.landmark[i].y = landmark_list.pose_landmarks.landmark[i].y * 1280
                frame_landmarks.update({ps: [landmark_list.pose_landmarks.landmark[i].x,
                                             landmark_list.pose_landmarks.landmark[i].y]})

        return frame_landmarks

    def __evaluate_data(self, data):
        """
        Prints model decision and cuts frame list (that thing in which data from frames are stored).
        :param data: list of dictionaries containing landmarks
        :return: data list with removed prefixed elements based on model decision
        """

        if len(data) < self.__model_frames:
            data_extended = self.__extend_data(data)
            df = pd.DataFrame(data_extended)
        else:
            df = pd.DataFrame(data)

        df = df.applymap(lambda x: round_list(x))

        tmp = np.array([row.tolist() for _, row in df.iterrows()])
        sampled = sample(tmp, self.__model_frames)
        tmp = ndarray_to_image(sampled)

        tmp = torch.from_numpy(tmp).type(torch.float32).to(self.__device)
        sampled = torch.from_numpy(sampled).type(torch.float32).to(self.__device)

        with torch.inference_mode():
            logit = self.__model(tmp)
            probs = torch.softmax(logit, dim=1)
            assigned_class_idx = probs.argmax(dim=1).item()
            assigned_class_name = self.__classes[assigned_class_idx]
            name = f"{self.__i}_{assigned_class_name.replace(' ', '_').lower()}_{int(torch.round(torch.max(probs) * 100).item())}"

            # visualisation_plot(tmp.cpu().numpy(), f"{name}.png", abspath(r"..\Visualisations\EvaluationPlots"))
            # visualisation_video(tmp.cpu().numpy(), f"{name}.avi", abspath(r"..\Visualisations\EvaluationVideos"))

            self.__i += 1

            # print(f"probabilities: {torch.round(probs * 100)}")

            if probs.max() > self.__finding_cutoff and assigned_class_idx == self.__current_exercise:
                self.__amount += 1;
                print(f"class: {assigned_class_name}, amount: {self.__amount}")
                plt.imsave(fr"Visualisations\EvaluationPlots\{name}.png", tmp.cpu().numpy())
                visualisation_video(sampled.cpu().numpy(), f"{name}.avi", abspath(r"Visualisations/EvaluationVideos"))
                return data[self.__cut_frames_find:]
            # else:
            #     print(f"class: NAH, I'd win")

        return data[self.__cut_frames_nothing:]

    def __eval_cam(self):
        """
        Evaluates camera input.
        :return: None
        """
        self.__model.eval()
        # mp_drawing = mp.solutions.drawing_utils
        mp_holistic = mp.solutions.holistic
        alldata = []
        cap = cv2.VideoCapture(0)

        with mp_holistic.Holistic(
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5) as holistic:

            while cap.isOpened():

                success, image = cap.read()

                if not success:
                    break

                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
                results = holistic.process(image)

                landmarks = self.__extract_landmarks(results)

                if landmarks:
                    alldata.append(landmarks)
                else:
                    continue

                if len(alldata) >= self.__frames_per_eval:
                    alldata = self.__evaluate_data(alldata)

                if cv2.waitKey(5) & 0xFF == 27:
                    break

        cap.release()

    def __eval_vid(self, video_path):
        """
        Evaluates video from specified path.
        :param video_path: video path :)
        :return: None
        """
        self.__model.eval()
        # mp_drawing = mp.solutions.drawing_utils
        mp_holistic = mp.solutions.holistic
        alldata = []
        cap = cv2.VideoCapture(video_path)

        with mp_holistic.Holistic(
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5) as holistic:
            while cap.isOpened():

                success, image = cap.read()

                if not success:
                    data_size = len(alldata)
                    if data_size < self.__model_frames:
                        print("Extending")
                        _ = self.__evaluate_data(self.__extend_data(alldata))

                    break

                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
                results = holistic.process(image)

                alldata.append(self.__extract_landmarks(results))

                if len(alldata) >= self.__frames_per_eval:
                    alldata = self.__evaluate_data(alldata)

                if cv2.waitKey(5) & 0xFF == 27:
                    break

        cap.release()

    def __extend_data(self, data):
        """
        Extends data in list if its size is too short for model input.
        :param data: list of landmarks
        :return: extended list of landmarks
        """
        multiplier = ceil(self.__model_frames / len(data))
        data_extended = [item for item in data for _ in range(multiplier)]
        return data_extended

    def run(self):
        """
        self-explanatory.
        :raises: ValueError
        :return: None
        """
        # I actually haven't tested it at all
        # what happens here remains to be seen
        # If it works then I am a god and if it doesn't then meh, who cares
        # Buggy's gonna become the KING OF THE PIRATES!!!!!!    TRUEEEEEEEEEEEEEE
        stop = False
        while not stop:
            cam_or_vid = None
            try:
                print("Menu\n"
                      "1. camera\n"  # still bad, but works (not as intended), at least it worked...
                      "2. video"
                      # info for future Joachim and Bubciu: ver. omega and further, doesn't work at all (at least it shouldn't)
                      )
                cam_or_vid = 1
                # cam_or_vid = int(input("Choice: "))

                print(
                    "\nExercises:\n"
                    "1: Jumping Jack\n"
                    "2: Side Leg Squat\n"
                    "3: Squat\n"
                    "4: Standing Sit-up\n"
                    "5: Side Bend\n"
                    "6: Bend\n"
                )

                exercise = int(input("Pick exercise: "))

                if exercise < 1 or exercise > 6:
                    raise ValueError("")

                self.__frames_per_eval = self.__frames_per_exercise[exercise][0]
                self.__cut_frames_find = self.__frames_per_exercise[exercise][1]
                self.__current_exercise = exercise

            except ValueError:
                print("You blind or stupid?")

            if cam_or_vid == 1:
                self.__eval_cam()
            elif cam_or_vid == 2:
                # file_path = input("Enter file path: ")
                # if os.path.exists(file_path):
                #
                #     self.__eval_vid(file_path)
                # else:
                #     print(f"File <{file_path}> does not exist")
                self.__eval_vid('SideScripts/ciąg_ćwiczeń.mp4')

            #stop = False if input("Continue?\nY/n: ").lower() in ["y", "yes", "yup", "yea", "yeah"] else True
            stop = True


if __name__ == "__main__":
    #deleting files from visualization folders
    VIS_PATH = Path(r'Visualisations')
    PLOTS_PATH = f'EvaluationPlots'
    VIDS_PATH = f'EvaluationVideos'
    delete_files_in_directory(VIS_PATH / PLOTS_PATH)
    delete_files_in_directory(VIS_PATH / VIDS_PATH)

    MODEL_LOAD_PATH = Path(r'Models')
    MODEL_LOAD_NAME = f'model_extended_stopido.pth'
    MODEL_LOAD_PATH = MODEL_LOAD_PATH / MODEL_LOAD_NAME
    tm = TestModel(7)
    tm.load_state_dict(torch.load(MODEL_LOAD_PATH))

    app = TestApp(tm)
    app.run()
