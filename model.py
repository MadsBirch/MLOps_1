import numpy as np
import torchvision
import torch, torch.nn as nn


class CNN(nn.Module):
    def __init__(self, in_channels = 1, num_classes = 10):
        super(CNN, self).__init__()
        self.conv = nn.Sequential(
                            nn.Conv2d(in_channels, 8, kernel_size = (3,3), stride=(1,1), padding=1),
                            nn.ReLU(inplace=True),
                            nn.MaxPool2d(kernel_size=(2,2), stride=(2,2)),
                            nn.Conv2d(8, 16, kernel_size = (3,3), stride=(1,1), padding=1),
                            nn.ReLU(inplace=True),
                            nn.MaxPool2d(kernel_size=(2,2), stride=(2,2)),
                            nn.Conv2d(16, 8, kernel_size = (3,3), stride=(1,1), padding=1),
                            nn.ReLU(inplace=True))
        self.fc = nn.Linear(8*7*7, num_classes)
    
    def forward(self, x):
        x = self.conv(x)
        x = torch.flatten(x,start_dim=1,end_dim=-1)
        x = self.fc(x)

        return x
