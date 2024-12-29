from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QPushButton, QListWidget, QListWidgetItem, QHBoxLayout, \
    QSpinBox, QGridLayout
from PyQt5.QtCore import Qt
from helper_functions import exercises_names


class WorkoutPlansWidget(QWidget):
    def __init__(self, back_button, exercise_list):
        super().__init__()
        self.exercise_list = exercise_list if exercise_list is not None else []

        self.layout = QVBoxLayout()

        self.label = QLabel("Workout Plans")
        self.layout.addWidget(self.label)

        self.list_widget = QListWidget()
        self.update_list_widget()

        self.remove_button = QPushButton("Remove Selected Exercise")
        self.remove_button.clicked.connect(self.remove_exercise)

        self.clear_button = QPushButton("Clear Workout Plan")
        self.clear_button.clicked.connect(self.clear_plan)

        self.add_exercise_layout = QGridLayout()
        self.add_exercise_layout.setAlignment(Qt.AlignTop)

        self.exercise_spinboxes = {}
        for i in range(1, 7):  # Exercises 1 to 6 (excluding 0)
            exercise_name = exercises_names[i][0]
            spinbox = QSpinBox()
            spinbox.setMinimum(1)
            spinbox.setValue(10)
            spinbox.setMaximum(100)
            add_button = QPushButton(f"Add {exercise_name}")
            add_button.clicked.connect(lambda _, idx=i, sb=spinbox: self.add_exercise(idx, sb.value()))

            self.add_exercise_layout.addWidget(QLabel(exercise_name), i - 1, 0)
            self.add_exercise_layout.addWidget(spinbox, i - 1, 1)
            self.add_exercise_layout.addWidget(add_button, i - 1, 2)

        self.layout.addWidget(self.list_widget)
        self.layout.addWidget(self.remove_button)
        self.layout.addWidget(self.clear_button)
        self.layout.addLayout(self.add_exercise_layout)
        self.layout.addWidget(back_button)

        self.setLayout(self.layout)


    def update_list_widget(self):
        self.list_widget.clear()
        for exercise in self.exercise_list:
            item_text = f"{exercises_names[exercise[0]][0]} x{exercise[1]}"
            item = QListWidgetItem(item_text)
            self.list_widget.addItem(item)


    def add_exercise(self, exercise_id, count):
        if len(self.exercise_list) != 0 and self.exercise_list[-1][0] == exercise_id:
            self.exercise_list[-1][1] += count
        else:  
            new_exercise = [exercise_id, count]
            self.exercise_list.append(new_exercise)
        self.update_list_widget()


    def remove_exercise(self):
        selected_item = self.list_widget.currentRow()
        if selected_item >= 0:
            del self.exercise_list[selected_item]
            self.update_list_widget()


    def clear_plan(self):
        self.exercise_list = []
        self.update_list_widget()


    def get_exercise_list(self):
        return self.exercise_list
