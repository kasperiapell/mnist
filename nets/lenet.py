# LeNet implementation in PyTorch

import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(
                        in_channels     = 1, 
                        out_channels    = 6, 
                        kernel_size     = 5, 
                        padding         = 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(5 * 5 * 16, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.avg_pool2d(
                x, 
                kernel_size    = 2, 
                stride         = 2)
        x = torch.sigmoid(self.conv2(x))
        x = F.avg_pool2d(x, 2, stride = 2)
        x = torch.flatten(x, 1)
        x = torch.sigmoid(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        x = self.fc3(x)

        return x
