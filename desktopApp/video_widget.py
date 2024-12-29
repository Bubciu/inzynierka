from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QPushButton, QFileDialog, QHBoxLayout, QSizePolicy, QComboBox
from PyQt5.QtGui import QPixmap, QImage, QFont
from PyQt5.QtCore import QThread, pyqtSignal as Signal, pyqtSlot as Slot
import cv2
import mediapipe as mp
from evaluation import ExerciseEvaluator, CorrectnessEvaluator
from helper_functions import extract_landmarks, exercises_dict, exercises_names

fps_mult = 0.0
process = ''
correctness_process = ''


class VideoThread(QThread):
    frame_signal = Signal(QImage)
    decision_signal = Signal(int)
    finished_signal = Signal()

    def __init__(self, video_path, current_exercise):
        super().__init__()
        self.video_path = video_path
        self._is_running = True
        self.cap = None
        self.current_exercise = current_exercise
        self.nedFrams = exercises_dict[current_exercise][0] * fps_mult
        self.exeval = ExerciseEvaluator(process)
        if current_exercise != 0:
            self.correctness_evaluator = CorrectnessEvaluator(current_exercise, correctness_process)
            

    def run(self):
        self.cap = cv2.VideoCapture(self.video_path)
        alldata = []
        mp_holistic = mp.solutions.holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)

        while self._is_running and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            label_frame = QImage(frame_rgb.data, frame_rgb.shape[1], frame_rgb.shape[0], QImage.Format_RGB888)
            self.frame_signal.emit(label_frame)

            results = mp_holistic.process(frame_rgb)
            landmarks = extract_landmarks(results)
            if landmarks:
                alldata.append(landmarks)

            if len(alldata) >= self.nedFrams:
                if self.current_exercise == 0:
                    model_ret = self.exeval.evaluate_data(alldata)
                    if model_ret != 0:
                        alldata = alldata[40:]
                        self.decision_signal.emit(model_ret)
                    else:
                        alldata = alldata[int(exercises_dict[0][1] * fps_mult):]
                else:
                    detected_exercise = self.exeval.evaluate_data(alldata, self.current_exercise)
                    if detected_exercise == self.current_exercise:
                        correctness = self.correctness_evaluator.evaluate_data(alldata)
                        self.decision_signal.emit(correctness)
                    alldata = alldata[int(exercises_dict[detected_exercise][1] * fps_mult):]

        self.cap.release()
        self.finished_signal.emit()

    def stop(self):
        self._is_running = False
        self.wait()


class VideoWidget(QWidget):
    def __init__(self, back_button, video_fps_mult, process_option, correctness_option):
        super().__init__()

        global fps_mult 
        fps_mult = video_fps_mult
        global process
        process = process_option
        global correctness_process
        correctness_process = correctness_option

        self.layout = QVBoxLayout()

        self.exercise_combo = QComboBox()
        self.exercise_combo.addItem("All", 0)
        for idx, name in list(exercises_names.items())[1:]:
            self.exercise_combo.addItem(name[0], idx)
        self.exercise_combo.currentIndexChanged.connect(self.change_exercise)
        self.layout.addWidget(self.exercise_combo)

        self.file_button = QPushButton("Choose File")
        self.file_button.clicked.connect(self.load_file)
        self.layout.addWidget(self.file_button)

        self.file_label = QLabel("Chosen File: None")
        self.file_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.layout.addWidget(self.file_label)

        self.hbox_layout = QHBoxLayout()

        self.video_label = QLabel()
        self.hbox_layout.addWidget(self.video_label)

        self.stats_layout = QVBoxLayout()
        self.stats_label = QLabel(
            "Stats\n\nJumping Jack: 0\nSide Leg Squat: 0\nSquat: 0\nStanding Sit-up: 0\nSide Bend: 0\nBend: 0")
        font = QFont()
        font.setPointSize(30)
        self.stats_label.setFont(font)
        self.stats_layout.addWidget(self.stats_label)
        self.hbox_layout.addLayout(self.stats_layout)

        self.layout.addLayout(self.hbox_layout)

        self.start_button = QPushButton("Start")
        self.start_button.clicked.connect(self.start_video)
        self.layout.addWidget(self.start_button)
        self.start_button.setEnabled(False)

        self.stop_button = QPushButton("Stop")
        self.stop_button.clicked.connect(self.stop_video)
        self.stop_button.setVisible(False)
        self.layout.addWidget(self.stop_button)

        self.back_button = back_button
        self.layout.addWidget(self.back_button)

        self.setLayout(self.layout)
        self.video_path = None
        self.video_thread = None
        self.current_exercise = 0
        self.correct_reps = 0
        self.incorrect_reps = 0
        self.exercise_counts = {
            "Jumping Jack": 0,
            "Side Leg Squat": 0,
            "Squat": 0,
            "Standing Sit-up": 0,
            "Side Bend": 0,
            "Bend": 0
        }


    def load_file(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self, "Wybierz plik wideo", "", "Wideo (*.mp4 *.avi)",
                                                   options=options)
        if file_path:
            self.video_path = file_path
            self.file_label.setText(f"Chosen File: {file_path.split('/')[-1]}")
            self.start_button.setEnabled(True)


    def start_video(self):
        if self.video_thread is not None:
            self.video_thread.stop()
        self.reset_stats()
        self.file_button.setEnabled(False)
        self.start_button.setVisible(False)
        self.stop_button.setVisible(True)
        self.exercise_combo.setEnabled(False)
        self.video_thread = VideoThread(self.video_path, self.current_exercise)
        self.video_thread.frame_signal.connect(self.update_frame)
        self.video_thread.decision_signal.connect(self.update_stats)
        self.video_thread.finished_signal.connect(self.on_video_finished)
        self.video_thread.start()


    def stop_video(self):
        if self.video_thread is not None:
            self.video_thread.stop()


    def reset_stats(self):
        self.correct_reps = 0
        self.incorrect_reps = 0
        for key in self.exercise_counts.keys():
            self.exercise_counts[key] = 0
        self.update_stats_label()


    def update_stats_label(self):
        if self.current_exercise == 0:
            stats_text = "\n".join([f"{name}: {count}" for name, count in self.exercise_counts.items()])
            self.stats_label.setText(f"Stats\n\n{stats_text}")
        else:
            self.stats_label.setText(f"Correct: {self.correct_reps}\nIncorrect: {self.incorrect_reps}")


    def change_exercise(self):
        new_exercise = self.exercise_combo.currentData()
        if new_exercise != self.current_exercise:
            self.current_exercise = new_exercise
        self.reset_stats()


    @Slot(QImage)
    def update_frame(self, image):
        self.video_label.setPixmap(QPixmap.fromImage(image))


    @Slot(int)
    def update_stats(self, decision):
        if self.current_exercise == 0:
            exercise_name = exercises_names[decision][0]
            self.exercise_counts[exercise_name] += 1
        else:
            if decision == 1:
                self.correct_reps += 1
            else:
                self.incorrect_reps += 1
        self.update_stats_label()


    @Slot()
    def on_video_finished(self):
        self.file_button.setEnabled(True)
        self.start_button.setVisible(True)
        self.stop_button.setVisible(False)
        self.exercise_combo.setEnabled(True)
        self.layout.update()


    def closeEvent(self, event):
        if self.video_thread is not None:
            self.video_thread.stop()
        event.accept()
