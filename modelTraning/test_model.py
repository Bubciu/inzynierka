import torch
from torch import nn


class TestModel(nn.Module):
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
        self.fc1 = nn.Linear(64 * 150 * 1, 128) #liczba kanałów * szerokość/2*100 * 1
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, num_classes)

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


class TestModelBinary(nn.Module):
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
