from PyQt5.QtWidgets import QWidget, QVBoxLayout, QPushButton, QApplication, QStackedLayout
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import QSize
from settings_widget import SettingsWidget
from live_camera_widget import LiveCameraWidget
from video_widget import VideoWidget
from workout_plans_widget import WorkoutPlansWidget
from correctness_widget import CorectnessWidget


class MainWidget(QWidget):
    def __init__(self, parent):
        super().__init__(parent)
        self.exercise_list = None
        self.layout = QVBoxLayout()

        self.stacked_layout = QStackedLayout()

        # settings
        self.cam_frame_mult = 1.0
        self.vid_frame_mult = 1.2

        # Initialize main buttons
        self.live_camera_button = QPushButton("Live Camera")
        self.live_camera_button.setIcon(QIcon("icons/videocam.svg"))
        self.live_camera_button.setIconSize(QSize(24, 24))
        self.live_camera_button.clicked.connect(self.show_live_camera)

        self.video_button = QPushButton("Video")
        self.video_button.setIcon(QIcon("icons/movie.svg"))
        self.video_button.setIconSize(QSize(24, 24))
        self.video_button.clicked.connect(self.show_video)

        self.corectness_button = QPushButton("Corectness")
        self.corectness_button.setIcon(QIcon("icons/add.svg"))
        self.corectness_button.setIconSize(QSize(24, 24))
        self.corectness_button.clicked.connect(self.show_corectness)

        self.workout_plans_button = QPushButton("Workout Plans")
        self.workout_plans_button.setIcon(QIcon("icons/exercise.svg"))
        self.workout_plans_button.setIconSize(QSize(24, 24))
        self.workout_plans_button.clicked.connect(self.show_workout_plans)

        self.settings_button = QPushButton("Settings")
        self.settings_button.setIcon(QIcon("icons/settings.svg"))
        self.settings_button.setIconSize(QSize(24, 24))
        self.settings_button.clicked.connect(self.show_settings)

        self.exit_button = QPushButton("Exit")
        self.exit_button.setIcon(QIcon("icons/Exit.svg"))
        self.exit_button.setIconSize(QSize(24, 24))
        self.exit_button.clicked.connect(self.exit_app)

        self.back_button = QPushButton("Back to Main")
        self.back_button.setIcon(QIcon("icons/arrow_back.svg"))
        self.back_button.setIconSize(QSize(24, 24))
        self.back_button.clicked.connect(self.show_main)
        self.back_button.hide()

        self.main_layout = QVBoxLayout()
        self.main_layout.addWidget(self.live_camera_button)
        self.main_layout.addWidget(self.video_button)
        self.main_layout.addWidget(self.corectness_button)
        self.main_layout.addWidget(self.workout_plans_button)
        self.main_layout.addWidget(self.settings_button)
        self.main_layout.addWidget(self.exit_button)
        self.main_layout.addWidget(self.back_button)

        self.main_widget = QWidget()
        self.main_widget.setLayout(self.main_layout)
        self.stacked_layout.addWidget(self.main_widget)

        self.setLayout(self.stacked_layout)

        self.live_camera_widget = None
        self.video_widget = None
        self.corectness_widget = None
        self.workout_plans_widget = None
        self.settings_widget = None


    def show_live_camera(self):
        self.live_camera_widget = LiveCameraWidget(self.back_button, self.exercise_list, self.cam_frame_mult)
        self.stacked_layout.addWidget(self.live_camera_widget)
        self.stacked_layout.setCurrentWidget(self.live_camera_widget)
        self.back_button.show()


    def show_video(self):
        self.video_widget = VideoWidget(self.back_button, self.vid_frame_mult)
        self.stacked_layout.addWidget(self.video_widget)
        self.stacked_layout.setCurrentWidget(self.video_widget)
        self.back_button.show()


    def show_corectness(self):
        self.corectness_widget = CorectnessWidget(self.back_button, self.cam_frame_mult)
        self.stacked_layout.addWidget(self.corectness_widget)
        self.stacked_layout.setCurrentWidget(self.corectness_widget)
        self.back_button.show()


    def show_workout_plans(self):
        self.workout_plans_widget = WorkoutPlansWidget(self.back_button, self.exercise_list)
        self.stacked_layout.addWidget(self.workout_plans_widget)
        self.stacked_layout.setCurrentWidget(self.workout_plans_widget)
        self.back_button.show()


    def show_settings(self):
        self.settings_widget = SettingsWidget(self.back_button)
        self.stacked_layout.addWidget(self.settings_widget)
        self.stacked_layout.setCurrentWidget(self.settings_widget)
        self.back_button.show()


    def show_main(self):
        if self.workout_plans_widget:
            self.exercise_list = self.workout_plans_widget.get_exercise_list()
        self.stacked_layout.setCurrentWidget(self.main_widget)
        self.back_button.hide()


    def update_cam_frame_mult(self, value):
        self.cam_frame_mult = value


    def update_vid_frame_mult(self, value):
        self.vid_frame_mult = value


    @staticmethod
    def exit_app():
        QApplication.instance().quit()
