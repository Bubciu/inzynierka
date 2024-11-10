from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QComboBox


class SettingsWidget(QWidget):
    def __init__(self, back_button):
        super().__init__()
        self.layout = QVBoxLayout()

        self.cam_frame_mult = 1.0
        self.vid_frame_mult = 1.2

        self.camera_fps_combo = QComboBox()
        self.camera_fps_combo.addItems(["20 FPS", "24 FPS", "30 FPS"])
        self.camera_fps_combo.setCurrentIndex(1)
        self.camera_fps_combo.currentIndexChanged.connect(self.update_camera_fps)

        # Combobox do wyboru FPS dla wideo
        self.video_fps_combo = QComboBox()
        self.video_fps_combo.addItems(["20 FPS", "24 FPS", "30 FPS"])
        self.video_fps_combo.setCurrentIndex(2)
        self.video_fps_combo.currentIndexChanged.connect(self.update_video_fps)

        self.label = QLabel("Settings")
        self.layout.addWidget(self.label)

        self.layout.addWidget(self.camera_fps_combo)
        self.layout.addWidget(self.video_fps_combo)

        self.layout.addWidget(back_button)
        self.setLayout(self.layout)


    def update_camera_fps(self, index):
        fps_options = {0: 0.8, 1: 1.0, 2: 1.2}
        self.cam_frame_mult = fps_options[index]
        self.parent().update_cam_frame_mult(self.cam_frame_mult)


    def update_video_fps(self, index):
        fps_options = {0: 0.8, 1: 1.0, 2: 1.2}
        self.vid_frame_mult = fps_options[index]
        self.parent().update_vid_frame_mult(self.vid_frame_mult)
