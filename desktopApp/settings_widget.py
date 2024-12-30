from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QComboBox, QFormLayout, QFrame
from PyQt5.QtCore import Qt

class SettingsWidget(QWidget):
    def __init__(self, back_button, parent):
        super().__init__()
        self.layout = QVBoxLayout()
        self.layout.setSpacing(15)

        self.setStyleSheet("QWidget { background-color: #f0f0f0; font-family: Arial, sans-serif; padding: 10px; }")

        header = QLabel("Settings")
        header.setAlignment(Qt.AlignCenter)
        header.setStyleSheet("font-size: 18px; font-weight: bold; color: #333;")

        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setFrameShadow(QFrame.Sunken)

        self.layout.addWidget(header)
        self.layout.addWidget(separator)

        self.processing_options = {0: 'unchanged', 1: 'plot', 2: 'trajectory'}
        default_exercise_process = [key for key, value in self.processing_options.items() if value == parent.process][0]
        default_correctness_process = [key for key, value in self.processing_options.items() if value == parent.correctness_process][0]

        self.fps_options = {0: 0.8, 1: 1.0, 2: 1.2}
        default_cam_fps = [key for key, value in self.fps_options.items() if value == parent.cam_frame_mult][0]
        default_vid_fps = [key for key, value in self.fps_options.items() if value == parent.vid_frame_mult][0]

        form_layout = QFormLayout()
        form_layout.setSpacing(10)

        self.process_combo = QComboBox()
        self.process_combo.addItems(["No Processing", "Plot", "Trajectory"])
        self.process_combo.setCurrentIndex(default_exercise_process)
        self.process_combo.setStyleSheet("QComboBox { background-color: white; border: 1px solid #ccc; border-radius: 5px; padding: 5px; font-size: 16px; }")
        self.process_combo.currentIndexChanged.connect(self.update_exercise_process)
        form_layout.addRow(QLabel("Exercise Processing:"), self.process_combo)

        self.correctness_process_combo = QComboBox()
        self.correctness_process_combo.addItems(["No Processing", "Plot", "Trajectory"])
        self.correctness_process_combo.setCurrentIndex(default_correctness_process)
        self.correctness_process_combo.setStyleSheet("QComboBox { background-color: white; border: 1px solid #ccc; border-radius: 5px; padding: 5px; font-size: 16px; }")
        self.correctness_process_combo.currentIndexChanged.connect(self.update_correctness_process)
        form_layout.addRow(QLabel("Correctness Processing:"), self.correctness_process_combo)

        self.camera_fps_combo = QComboBox()
        self.camera_fps_combo.addItems(["20 FPS", "24 FPS", "30 FPS"])
        self.camera_fps_combo.setCurrentIndex(default_cam_fps)
        self.camera_fps_combo.setStyleSheet("QComboBox { background-color: white; border: 1px solid #ccc; border-radius: 5px; padding: 5px; font-size: 16px; }")
        self.camera_fps_combo.currentIndexChanged.connect(self.update_camera_fps)
        form_layout.addRow(QLabel("Camera FPS:"), self.camera_fps_combo)

        self.video_fps_combo = QComboBox()
        self.video_fps_combo.addItems(["20 FPS", "24 FPS", "30 FPS"])
        self.video_fps_combo.setCurrentIndex(default_vid_fps)
        self.video_fps_combo.setStyleSheet("QComboBox { background-color: white; border: 1px solid #ccc; border-radius: 5px; padding: 5px; font-size: 16px; }")
        self.video_fps_combo.currentIndexChanged.connect(self.update_video_fps)
        form_layout.addRow(QLabel("Video FPS:"), self.video_fps_combo)

        self.layout.addLayout(form_layout)

        back_button.setStyleSheet("QPushButton { background-color: #0078d7; color: white; border: none; border-radius: 5px; padding: 10px; font-size: 16px; margin: 5px; } QPushButton:hover { background-color: #0056a1; }")
        self.layout.addWidget(back_button, alignment=Qt.AlignCenter)

        self.setLayout(self.layout)


    def update_exercise_process(self, index):
        self.parent().set_process(self.processing_options[index])


    def update_correctness_process(self, index):
        self.parent().set_correctness_process(self.processing_options[index])


    def update_camera_fps(self, index):
        self.parent().set_cam_frame_mult(self.fps_options[index])


    def update_video_fps(self, index):
        self.parent().set_vid_frame_mult(self.fps_options[index])
