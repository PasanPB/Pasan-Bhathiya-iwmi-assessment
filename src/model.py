import torch
import torch.nn as nn
import torch.nn.functional as F


class ModelDevelopment:

    def __init__(self):
        self.model = self.build_model()

    def build_model(self):
        return MaskCNN()

    def get_model(self):
        return self.model

    def any_name(self):
        print("Custom model method placeholder")


class MaskCNN(nn.Module):
    def __init__(self):
        super(MaskCNN, self).__init__()

        # 🔹 Block 1
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        # 🔹 Block 2
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        # 🔹 Block 3
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)

        # 🔹 Fully Connected
        self.fc1 = nn.Linear(128 * 16 * 16, 128)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))

        x = x.view(x.size(0), -1)

        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)

        return x


class MaskClassifier(MaskCNN):
    pass