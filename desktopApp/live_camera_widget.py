import cv2
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QPushButton, QHBoxLayout, QSizePolicy
from PyQt5.QtGui import QPixmap, QImage, QFont
from PyQt5.QtCore import QThread, pyqtSignal as Signal, pyqtSlot as Slot
from evaluation import *
import mediapipe as mp
from helper_functions import extract_landmarks, exercises_dict, exercises_names

fps_mult = 0.0
process = ''


class MyThread(QThread):
    frame_signal = Signal(QImage)
    decision_signal = Signal(int)

    def __init__(self, current_exercise):
        super().__init__()
        self._is_running = True
        self.cap = None
        self.nedFrams = exercises_dict[0][0] * fps_mult
        self.current_exercise = current_exercise
        self.exeval = ExerciseEvaluator(process)

    def run(self):
        alldata = []
        mp_holistic = mp.solutions.holistic
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        with mp_holistic.Holistic(
                min_detection_confidence=0.3,
                min_tracking_confidence=0.3) as holistic:

            while self._is_running and self.cap.isOpened():
                ret, frame = self.cap.read()
                if not ret:
                    break

                label_frame = self.cvimage_to_label(frame)
                self.frame_signal.emit(label_frame)

                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame.flags.writeable = False
                results = holistic.process(frame)

                landmarks = extract_landmarks(results)

                # jeśli istnieją, dodaj do zbioru klatek
                if landmarks:
                    alldata.append(landmarks)
                else:
                    continue

                # jeśli mamy wystarczająco klatek, wyślij je do ewaluacji
                if len(alldata) >= self.nedFrams:
                    model_ret = self.exeval.evaluate_data(alldata,
                                                     self.current_exercise if self.current_exercise != 0 else None)
                    alldata = alldata[int(exercises_dict[model_ret][1] * fps_mult):]

                    if model_ret != 0:
                        self.decision_signal.emit(model_ret)

        self.cap.release()

    def stop(self):
        self._is_running = False
        self.wait()

    def update_exercise(self, new_exercise):
        self.current_exercise = new_exercise
        self.nedFrams = exercises_dict[new_exercise][0] * fps_mult

    @staticmethod
    def cvimage_to_label(image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = QImage(image,
                       image.shape[1],
                       image.shape[0],
                       QImage.Format_RGB888)
        return image


class LiveCameraWidget(QWidget):
    def __init__(self, back_button, exercise_list=None, camera_fps_mult=0.0, process_option=''):
        super().__init__()

        global fps_mult
        fps_mult = camera_fps_mult
        global process
        process = process_option

        self.exercise_reps_done = 0
        self.exercise_list = exercise_list
        self.exercise_idx = 0
        self.exercise_counts = {
            "Jumping Jack": 0,
            "Side Leg Squat": 0,
            "Squat": 0,
            "Standing Sit-up": 0,
            "Side Bend": 0,
            "Bend": 0
        }

        if not self.exercise_list:
            self.current_exercise = 0
            self.exercise_reps_to_do = 0
        else:
            self.current_exercise = exercise_list[self.exercise_idx][0]
            self.exercise_reps_to_do = exercise_list[self.exercise_idx][1]

        self.layout = QVBoxLayout()

        self.camera_and_score_layout = QHBoxLayout()

        self.camera_label = QLabel()
        self.camera_and_score_layout.addWidget(self.camera_label)

        self.score_label = QLabel()
        self.score_label.setFont(QFont("Arial", 50))
        self.camera_and_score_layout.addWidget(self.score_label)

        self.layout.addLayout(self.camera_and_score_layout)

        self.open_btn = QPushButton("Open The Camera")
        self.open_btn.clicked.connect(self.open_camera)
        self.layout.addWidget(self.open_btn)

        self.close_btn = QPushButton("Close The Camera")
        self.close_btn.clicked.connect(self.close_camera)
        self.layout.addWidget(self.close_btn)
        self.close_btn.hide()

        self.camera_thread = None

        self.button_layout = QHBoxLayout()
        self.button_layout.addWidget(back_button)
        self.layout.addLayout(self.button_layout)

        self.setLayout(self.layout)

    def open_camera(self):
        if self.camera_thread is None or not self.camera_thread.isRunning():
            self.camera_thread = MyThread(self.current_exercise)

            if self.current_exercise == 0:
                stats_text = "\n".join([f"{name}: {count}" for name, count in self.exercise_counts.items()])
                self.score_label.setText(f"Stats\n\n{stats_text}")
            else:
                self.score_label.setText(f"{exercises_names[self.current_exercise][0]}: "
                                                    f"{self.exercise_reps_done}/{self.exercise_reps_to_do}")
            
            self.score_label.adjustSize()

            self.camera_thread.frame_signal.connect(self.setImage)
            self.camera_thread.decision_signal.connect(self.showDecision)
            self.camera_thread.start()
            self.close_btn.show()
            self.open_btn.hide()

    def close_camera(self):
        if self.camera_thread is not None:
            self.camera_thread.stop()
            self.camera_thread = None
            self.close_btn.hide()
            self.open_btn.show()

    def update_thread_exercise(self, new_exercise):
        if self.camera_thread is not None:
            self.camera_thread.update_exercise(new_exercise)

    @Slot(QImage)
    def setImage(self, image):
        if self.camera_thread is not None:
            self.camera_label.setPixmap(QPixmap.fromImage(image))

    @Slot(int)
    def showDecision(self, decision):
        if self.camera_thread is None:
            return
        
        if self.current_exercise == 0:
            exercise_name = exercises_names[decision][0]
            self.exercise_counts[exercise_name] += 1
            stats_text = "\n".join([f"{name}: {count}" for name, count in self.exercise_counts.items()])
            self.score_label.setText(f"Stats\n\n{stats_text}")
            
        elif decision == self.current_exercise:
            self.exercise_reps_done += 1
            if self.exercise_reps_done >= self.exercise_reps_to_do:
                self.exercise_idx += 1
                if self.exercise_idx == len(self.exercise_list):
                    self.current_exercise = 0
                    self.update_thread_exercise(self.current_exercise)
                    return
                else:
                    self.current_exercise = self.exercise_list[self.exercise_idx][0]
                    self.update_thread_exercise(self.current_exercise)
                    self.exercise_reps_to_do = self.exercise_list[self.exercise_idx][1]
                    self.exercise_reps_done = 0

            self.score_label.setText(f"{exercises_names[self.current_exercise][0]}: "
                                        f"{self.exercise_reps_done}/{self.exercise_reps_to_do}")
            
        self.score_label.adjustSize()
