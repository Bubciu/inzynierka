import sys
import warnings

from PyQt5.QtWidgets import QApplication, QMainWindow
from main_widget import MainWidget


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Workout App")
        self.setGeometry(0, 0, 1920, 1080)

        self.main_widget = MainWidget(self)
        self.setCentralWidget(self.main_widget)

    def show(self):
        super().showMaximized()


warnings.filterwarnings("ignore")

if __name__ == "__main__":
    app = QApplication(sys.argv)

    with open("desktopApp/style/styles.qss", "r") as f:
        app.setStyleSheet(f.read())

    mainWin = MainWindow()
    mainWin.show()

    sys.exit(app.exec_())
