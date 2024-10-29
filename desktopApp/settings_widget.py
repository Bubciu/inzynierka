from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel


class SettingsWidget(QWidget):
    def __init__(self, back_button):
        super().__init__()
        self.layout = QVBoxLayout()
        self.label = QLabel("Settings")
        self.layout.addWidget(self.label)
        self.layout.addWidget(back_button)
        self.setLayout(self.layout)
