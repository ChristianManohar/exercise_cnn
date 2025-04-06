"""
Defines the architecture for a Convolutional Neural Network
Uses architecture from Guo et al. defined in this paper: 
https://ieeexplore.ieee.org/abstract/document/8078730/figures#figures

Architecture:
    Convolutional Layer 1:
        num_output = 64
        kernel_size = 5
        stride = 1
        padding = 2
        ReLU activation
    Max Pool:
        Kernel_size = 2
        stride = 2
    Convolutional Layer 2:
        Same as conv1
    Max Pool
    BatchNorm
    Conv 3
    Max Pool
    BatchNorm
    FC1:
        input: shape * 64 inputs
        output: 500
        relu
        dropout - p = 0.5
    FC2:
        input = 500
        output = 17 (17 categories)
"""

import torch
import torch.nn as nn
import torch.nn.functional as func
from math import sqrt

class CNN(nn.Module):
    def __init__(self):
        #Define layers of CNN
        super().__init__()
        self.kernel_size = 5
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5, stride=1, padding=2)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, stride=1, padding=2)
        self.norm = nn.BatchNorm2d(num_features = 64)
        self.conv3 = nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size=5, stride=1, padding=2)
        self.fc1 = nn.Linear(in_features = 4096, out_features = 500)
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(in_features = 500, out_features = 22)

        self.set_weights()
    
    def set_weights(self):
        #Initialize weights using normal and He initialization
        for conv in [self.conv1, self.conv2, self.conv3]:
            nn.init.normal_(conv.weight, mean = 0, std = sqrt(2 / ((self.kernel_size**2) * conv.in_channels)))
            nn.init.constant_(conv.bias, float(1.0))
            pass
        for fc in [self.fc1, self.fc2]:
            nn.init.kaiming_normal_(fc.weight, mode="fan_in")
            nn.init.constant_(fc.bias, float(0.0))
            pass
    
    def forward(self, x):
        #Convolutional layer 1
        #print(x.shape)
        x = func.relu(self.conv1(x))
        #print(x.shape)
        #Max Pooling layer
        x = self.pool(x)
        #print(x.shape)

        #Convolutional Layer 2
        x = func.relu(self.conv2(x))
        #print(x.shape)
        #Max Pooling and BatchNorm layer
        x = self.pool(x)
        #print(x.shape)
        x = self.norm(x)
        #print(x.shape)

        #Convolutional layer 3
        x = func.relu(self.conv3(x))
        #print(x.shape)

        x = self.pool(x)
        #print(x.shape)
        x = self.norm(x)
        #print(x.shape, 1)

        #Reshape for both fully connected layers
        x = torch.flatten(x, 1)
        #print(x.shape, 1)
        x = func.relu(self.fc1(x))
        #print(x.shape)

        x = self.dropout(x)

        x = self.fc2(x)
        return x
    