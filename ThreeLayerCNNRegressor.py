import torch.nn as nn
import torch

class ThreeLayerCNNRegressor(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, 3)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(10, 20, 5)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(2)
        self.conv3 = nn.Conv2d(20, 20, 7)
        self.relu3 = nn.ReLU()
        self.maxpool3 = nn.MaxPool2d(2)
        self.fc1 = nn.LazyLinear(100)
        self.relu4 = nn.ReLU()
        self.fc2 = nn.Linear(100, 10)
        self.relu5 = nn.ReLU()
        self.fc3 = nn.Linear(10, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.maxpool3(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.relu4(x)
        x = self.fc2(x)
        x = self.relu5(x)
        x = self.fc3(x)
        return x
