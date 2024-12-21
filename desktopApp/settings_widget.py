from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QComboBox


class SettingsWidget(QWidget):
    def __init__(self, back_button, parent):
        super().__init__()
        self.layout = QVBoxLayout()

        self.processing_options = {0: 'unchanged', 1: 'plot', 2: 'trajectory'}

        default_process = [key for key, value in self.processing_options.items() if value == parent.process][0]

        self.fps_options = {0: 0.8, 1: 1.0, 2: 1.2}

        default_cam_fps = [key for key, value in self.fps_options.items() if value == parent.cam_frame_mult][0]
        default_vid_fps = [key for key, value in self.fps_options.items() if value == parent.vid_frame_mult][0]

        self.process_combo = QComboBox()
        self.process_combo.addItems(["No Processing", "Plot", "Trajectory"])
        self.process_combo.setCurrentIndex(default_process)
        self.process_combo.currentIndexChanged.connect(self.update_process)

        self.camera_fps_combo = QComboBox()
        self.camera_fps_combo.addItems(["20 FPS", "24 FPS", "30 FPS"])
        self.camera_fps_combo.setCurrentIndex(default_cam_fps)
        self.camera_fps_combo.currentIndexChanged.connect(self.update_camera_fps)

        self.video_fps_combo = QComboBox()
        self.video_fps_combo.addItems(["20 FPS", "24 FPS", "30 FPS"])
        self.video_fps_combo.setCurrentIndex(default_vid_fps)
        self.video_fps_combo.currentIndexChanged.connect(self.update_video_fps)

        self.label = QLabel("Settings")
        self.layout.addWidget(self.label)

        self.layout.addWidget(self.process_combo)
        self.layout.addWidget(self.camera_fps_combo)
        self.layout.addWidget(self.video_fps_combo)

        self.layout.addWidget(back_button)
        self.setLayout(self.layout)

    def update_process(self, index):
        self.parent().set_process(self.processing_options[index])


    def update_camera_fps(self, index):
        self.parent().set_cam_frame_mult(self.fps_options[index])


    def update_video_fps(self, index):
        self.parent().set_vid_frame_mult(self.fps_options[index])
