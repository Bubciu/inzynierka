import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt


class ExerciseEvaluationModel(nn.Module):
    """
    Neural network class.
    This class defines a convolutional neural network with two convolutional layers followed by
    fully connected layers for classification tasks.
    """

    def __init__(self, num_classes):
        """
        Initialiser. Sets every layer of nn.
        :param num_classes: number of model output classes
        """

        super().__init__()

        # 2D Convolutional layers
        self.conv1 = nn.Conv2d(300, 32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=1, stride=1)

        # Fully connected layers
        self.fc1 = nn.Linear(64 * 150 * 1, 128)     # conv2-n_channels * (img_width/2 * 100) * 1
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        """
        Defines the forward pass of the model.

        :param x: Input tensor.
        :return: Output tensor - probabilities of each class.
        """

        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)

        x = x.view(-1, 64 * 150 * 1)                            # conv2-n_channels * (img_width/2 * 100) * 1
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)

        return x


class CorectnessEvaluationModel(nn.Module):
    """
    Neural network class.
    This class defines a convolutional neural network with two convolutional layers followed by
    fully connected layers for classification tasks.
    """

    def __init__(self):
        """
        Initialiser. Sets every layer of nn.
        :param num_classes: number of model output classes
        """

        super().__init__()

        # 2D Convolutional layers
        self.conv1 = nn.Conv2d(300, 32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=1, stride=1)

        # Fully connected layers
        self.fc1 = nn.Linear(64 * 150 * 1, 128) #liczba kanałów * szerokość/2*100 * 1
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        """
        Forward pass of the neural network.
        :param x: Input tensor of shape (batch_size, 300, height, width)
        :return: Output tensor of shape (batch_size, num_classes)
        """

        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)

        x = x.view(-1, 64 * 150 * 1)        # conv2-n_channels * (img_width/2 * 100) * 1
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)

        return x


def sample(data: np.ndarray, number_of_samples: int) -> np.ndarray:
    """
    Samples provided ndarray
    :param data: ndarray with landmark coordinates
    :param number_of_samples: number of samples to take from data
    :return: sampled ndarray
    """
    return data[np.sort(np.random.choice(data.shape[0], number_of_samples, False))]


def ndarray_to_image(landmarks: np.ndarray, show: bool = False) -> np.ndarray:
    """
    Function creates image (.png) from provided array containing landmarks' x and y coordinates.
    :param landmarks: 3-dimensional numpy array with landmarks' coordinates in frames
    :param show: Specifies if the image should be shown.
    :return: Image with pixel colour in rage <0, 1>.
    """

    fig, axs = plt.subplots(1, 2, figsize=(3, 3))
    plt.tight_layout(pad=0)

    axs[0].get_xaxis().set_visible(False)
    axs[0].get_yaxis().set_visible(False)
    axs[0].spines['top'].set_visible(False)
    axs[0].spines['right'].set_visible(False)
    axs[0].spines['bottom'].set_visible(False)
    axs[0].spines['left'].set_visible(False)

    axs[1].get_xaxis().set_visible(False)
    axs[1].get_yaxis().set_visible(False)
    axs[1].spines['top'].set_visible(False)
    axs[1].spines['right'].set_visible(False)
    axs[1].spines['bottom'].set_visible(False)
    axs[1].spines['left'].set_visible(False)

    axs[0].imshow(landmarks[:, :, 0])
    axs[1].imshow(landmarks[:, :, 1])

    canvas = fig.canvas

    canvas.draw()

    image_flat = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')  # (H * W * 3,)
    image = image_flat.reshape(*reversed(canvas.get_width_height()), 3)  # (H, W, 3)

    if show:
        # cv2.imshow('win', image)
        # cv2.waitKey(0)
        plt.show()
    else:
        fig.clear()
        plt.close()

    return image / 255.0


def round_list(values: list) -> list:
    """
    Rounds each element in the list to 3 decimal places.

    :param values: List of numbers.
    :return: List with rounded numbers.
    """

    return [round(x, 3) for x in values]
