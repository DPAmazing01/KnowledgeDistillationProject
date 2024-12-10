import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        if self.downsample is not None:
            identity = self.downsample(identity)
        x += identity
        x = self.relu(x)
        return x

class ResNet18(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet18, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = nn.Sequential(
            BasicBlock(64, 64),
            BasicBlock(64, 64),
        )
        self.layer2 = nn.Sequential(
            BasicBlock(64, 128, 2, nn.Sequential(
                nn.Conv2d(64, 128, 1, 2, bias=False),
                nn.BatchNorm2d(128)
            )),
            BasicBlock(128, 128),
        )
        self.layer3 = nn.Sequential(
            BasicBlock(128, 256, 2, nn.Sequential(
                nn.Conv2d(128, 256, 1, 2, bias=False),
                nn.BatchNorm2d(256)
            )),
            BasicBlock(256, 256),
        )
        self.layer4 = nn.Sequential(
            BasicBlock(256, 512, 2, nn.Sequential(
                nn.Conv2d(256, 512, 1, 2, bias=False),
                nn.BatchNorm2d(512)
            )),
            BasicBlock(512, 512),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

class ResNet18_half(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet18_half, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = nn.Sequential(
            BasicBlock(32, 32),
            BasicBlock(32, 32),
        )
        self.layer2 = nn.Sequential(
            BasicBlock(32, 64, 2, nn.Sequential(
                nn.Conv2d(32, 64, 1, 2, bias=False),
                nn.BatchNorm2d(64)
            )),
            BasicBlock(64, 64),
        )
        self.layer3 = nn.Sequential(
            BasicBlock(64, 128, 2, nn.Sequential(
                nn.Conv2d(64, 128, 1, 2, bias=False),
                nn.BatchNorm2d(128)
            )),
            BasicBlock(128, 128),
        )
        self.layer4 = nn.Sequential(
            BasicBlock(128, 256, 2, nn.Sequential(
                nn.Conv2d(128, 256, 1, 2, bias=False),
                nn.BatchNorm2d(256)
            )),
            BasicBlock(256, 256),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, num_classes)    

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
