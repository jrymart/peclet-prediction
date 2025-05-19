import torch.nn as nn
import torch

RIDGE_WEIGHTS = [
    torch.tensor([[[0,1,0],
                  [0,1,0],
                  [0,1,0]]]),
    torch.tensor([[[0,0,0],
                  [1,1,1],
                  [0,0,0]]]),
    torch.tensor([[[1,0,0],
                  [0,1,0],
                  [0,0,1]]]),
    torch.tensor([[[0,0,1],
                  [0,1,0],
                  [1,0,0]]]),
]

FACE_WEIGHTS = [
    torch.tensor([[[1,   0.5,  0],
                  [0.5, 0,   -0.5],
                  [0,  -0.5, -1]]]),
    torch.tensor([[[0,   0.5,  1],
                  [-0.5, 0,   0.5],
                  [-1,  -0.5, 0]]]),
    torch.tensor([[[-1,   -0.5,  0],
                  [-0.5, 0,   0.5],
                  [0,  0.5, 1]]]),
    torch.tensor([[[0,   -0.5, -1],
                  [0.5, 0,   -0.5],
                  [1,  0.5, 0]]]),
    torch.tensor([[[1, 1, 1],
                  [0, 0, 0],
                  [-1, -1, -1]]]),
    torch.tensor([[[-1, -1, -1],
                  [0, 0, 0],
                  [1, 1, 1]]]),
    torch.tensor([[[-1, 0, 1],
                  [-1, 0, 1],
                  [-1, 0, 1]]]),
    torch.tensor([[[1, 0, -1],
                  [1, 0, -1],
                  [1, 0, -1]]])
    ]

class RidgeFaceCNN(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 12, kernel_size=3)
        self.conv2 = nn.Conv2d(12, 20, kernel_size=5)
        self.conv3 = nn.Conv2d(20, 20, kernel_size=7)
        self.pools = nn.ModuleList([nn.MaxPool2d(kernel_size=2, stride=2) for _ in range(3)])
        self.relus = nn.ModuleList([nn.ReLU() for _ in range(5)])
        self.fc1 = nn.LazyLinear(100)
        self.fc2 = nn.Linear(100, 10)
        self.fc3 = nn.Linear(10, 1)

    def _initialize_conv1_weights(self):
        combined_weights = torch.cat(RIDGE_WEIGHTS+FACE_WEIGHTS, dim=0)
        expected_shape = self.conv1.weight.shape
        if combined_weights.shape != expected_shape:
            raise ValueError(f"Expected shape {expected_shape}, but got {combined_weights.shape}")
        with torch.no_grad():
            self.conv1.weight.data.copy_(combined_weights)
            if self.conv1.bias is not None:
                self.conv1.bias.data.zero_()
                self.conv1.bias.requires_grad = False
        self.conv1.weight.requires_grad = False

    def forward(self, x):
        x = self.conv1(x)
        x = self.pools[0](x)
        x = self.relus[0](x)
        x = self.conv2(x)
        x = self.pools[1](x)
        x = self.relus[1](x)
        x = self.conv3(x)
        x = self.pools[2](x)
        x = self.relus[2](x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.relus[3](x)
        x = self.fc2(x)
        x = self.relus[4](x)
        x = self.fc3(x)
        return x
