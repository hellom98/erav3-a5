import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()

        self.pool1 = nn.MaxPool2d(2)
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(8, affine=False)

        self.pool2 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(8, 64, kernel_size=3, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(64, affine=False)
        self.fc = nn.Linear(64 * 16, 10, bias=False)
        
    def forward(self, x):
        # x: [batch_size, 1, 28, 28]
        x = self.pool1(x) # 14x14
        x = self.conv1(x) # 12x12
        x = F.relu(x)
        x = self.bn1(x) 

        x = self.pool2(x) # 6x6
        x = self.conv2(x) # 4x4
        x = F.relu(x)
        x = self.bn2(x)

        x = x.view(-1, 64 * 4 * 4) # 64 * 16
        
        x = self.fc(x)
        return x