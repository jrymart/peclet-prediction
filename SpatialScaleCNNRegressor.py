import torch.nn as nn
import torch
import numpy as np

class SpatialScaleCNNRegressor(nn.Module):

    def __init__(self, scales=np.arange(3,21), out_convs=3, padding_mode='zeros', aggregate_scale=10):
        super().__init__()
        self.scale_layers = nn.ModuleList([nn.Conv2d(1,out_convs,i, padding='same',padding_mode=padding_mode) for i in scales])
        self.scale_pools = nn.ModuleList([nn.MaxPool2d(2) for _ in scales])
        self.scale_relus = nn.ModuleList([nn.ReLU() for _ in scales])
        output_layers = out_convs*len(self.scale_layers)
        self.final_aggregate = nn.Conv2d(output_layers, 10, aggregate_scale)
        self.final_pool = nn.MaxPool2d(2)
        self.final_conv_relu = nn.ReLU()
        self.fc1 = nn.LazyLinear(100)
        self.relu2 = nn.ReLU()
        self.fc2 = nn.Linear(100,10)
        self.relu3 = nn.ReLU()
        self.fc3 = nn.Linear(10,1)

    def forward(self,x):
        scale_feature_maps = []
        for i, conv in enumerate(self.scale_layers):
            xi = conv(x)
            xi = self.scale_pools[i](xi)
            xi = self.scale_relus[i](xi)
            scale_feature_maps.append(xi)
        x = torch.cat(scale_feature_maps, dim=1)
        x = self.final_aggregate(x)
        x = self.final_pool(x)
        x = self.final_conv_relu(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.relu2(x)
        x = self.fc2(x)
        x = self.relu3(x)
        x = self.fc3(x)
        return x
