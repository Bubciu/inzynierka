from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QPushButton, QHBoxLayout, QSizePolicy, QComboBox
from PyQt5.QtGui import QPixmap, QImage, QFont
from PyQt5.QtCore import QThread, pyqtSignal as Signal, pyqtSlot as Slot
import cv2
import mediapipe as mp

from evaluation import *
from helper_functions import extract_landmarks, exercises_dict, exercises_names
from evaluation import *

fps_mult = 0.0

class MyThread(QThread):
    frame_signal = Signal(QImage)
    decision_signal = Signal(int)  # Zmiana: wysyłamy wynik wykrycia i poprawności

    def __init__(self, current_exercise):
        super().__init__()
        self._is_running = True
        self.cap = None
        self.current_exercise = current_exercise
        self.nedFrams = exercises_dict[current_exercise][0]
        self.exercise_evaluator = ExerciseEvaluator()  # Inicjalizacja evaluatora ćwiczeń
        self.correctness_evaluator = CorrectnessEvaluator(current_exercise)  # Inicjalizacja evaluatora poprawności

    def run(self):
        alldata = []
        mp_holistic = mp.solutions.holistic
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        with mp_holistic.Holistic(
            min_detection_confidence=0.5, 
            min_tracking_confidence=0.5) as holistic:

            while self._is_running and self.cap.isOpened():
                ret, frame = self.cap.read()
                if not ret:
                    break
                
                label_frame = self.cvimage_to_label(frame)
                self.frame_signal.emit(label_frame)

                # mediapipe processing
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame.flags.writeable = False
                results = holistic.process(frame)

                landmarks = extract_landmarks(results)
                if landmarks:
                    alldata.append(landmarks)
                else:
                    continue

                if len(alldata) >= self.nedFrams:
                    detected_exercise = self.exercise_evaluator.evaluate_data(alldata, self.current_exercise)

                    if detected_exercise == self.current_exercise:
                        correctness = self.correctness_evaluator.evaluate_data(alldata)
                        self.decision_signal.emit(correctness)  # Emitujemy wynik wykrycia i poprawności

                    alldata = alldata[exercises_dict[detected_exercise][1]:]

        self.cap.release()

    def stop(self):
        self._is_running = False
        self.wait()

    def update_exercise(self, new_exercise):
        self.current_exercise = new_exercise
        self.nedFrams = exercises_dict[new_exercise][0]
        self.correctness_evaluator = CorrectnessEvaluator(new_exercise)  # Aktualizacja evaluatora poprawności

    @staticmethod
    def cvimage_to_label(image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return QImage(image, image.shape[1], image.shape[0], QImage.Format_RGB888)


class CorectnessWidget(QWidget):
    def __init__(self, back_button, camera_fps_mult):
        super().__init__()

        fps_mult = camera_fps_mult

        self.correct_reps = 0
        self.incorrect_reps = 0
        self.current_exercise = 1

        self.layout = QVBoxLayout()

        # Wybór ćwiczenia
        self.exercise_combo = QComboBox()
        for idx, name in list(exercises_names.items())[1:]:
            self.exercise_combo.addItem(name[0], idx)
        self.exercise_combo.currentIndexChanged.connect(self.change_exercise)
        self.layout.addWidget(self.exercise_combo)

        # Układ dla kamery i wyników
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

        # Układ dla przycisku powrotu
        self.button_layout = QHBoxLayout()
        self.button_layout.addWidget(back_button)
        self.layout.addLayout(self.button_layout)

        self.setLayout(self.layout)

    def open_camera(self):
        if self.camera_thread is None or not self.camera_thread.isRunning():
            self.camera_thread = MyThread(self.current_exercise)
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

    def change_exercise(self):
        new_exercise = self.exercise_combo.currentData()
        if new_exercise != self.current_exercise:
            self.current_exercise = new_exercise
            if self.camera_thread is not None:
                self.camera_thread.update_exercise(new_exercise)

    @Slot(QImage)
    def setImage(self, image):
        if self.camera_thread is not None:
            self.camera_label.setPixmap(QPixmap.fromImage(image))

    @Slot(int)
    def showDecision(self, correctness):
        if correctness == 1:
            self.correct_reps += 1
        else:
            self.incorrect_reps += 1

        self.score_label.setText(f"Correct: {self.correct_reps}\nIncorrect: {self.incorrect_reps}")
        self.score_label.adjustSize()
