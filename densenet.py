import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

class Transition(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(Transition, self).__init__()
        self.norm = nn.BatchNorm2d(in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
    
    def forward(self, x):
        out = self.norm(x)
        out = self.conv(out)
        out = self.pool(out)
        return out

class DenseLayer(nn.Module):
    def __init__(self, in_channels, growth_rate, bn_size):
        super(DenseLayer, self).__init__()
        self.norm1 = nn.BatchNorm2d(in_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, bn_size * growth_rate, kernel_size=1, stride=1, bias=False)

        self.norm2 = nn.BatchNorm2d(bn_size * growth_rate)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(bn_size * growth_rate, growth_rate, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x):
        # Bottleneck
        prev_in = x
        out = self.norm1(x)
        out = self.relu1(out)
        out = self.conv1(out)

        # Forward
        out = self.norm2(out)
        out = self.relu2(out)
        out = self.conv2(out)

        out = torch.cat([prev_in, out], 1)
        return out
    
class DenseBlock(nn.ModuleDict):
    def __init__(self, num_layers, in_channels, bn_size, growth_rate):
        super(DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = DenseLayer(in_channels + i * growth_rate, growth_rate=growth_rate, bn_size=bn_size)
            self.add_module('denselayer%d' % (i + 1), layer)

    def forward(self, x):
        out = x
        for name, layer in self.items():
            out = layer(out)
        return out

class DenseNet(nn.Module):
    def __init__(self):
        super(DenseNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 7, 2, 3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(3, 2, 1)

        self.block1 = DenseBlock(6, 64, 4, 32)
        self.transition1 = Transition(64+32*6, 128)

        self.block2 = DenseBlock(12, 128, 4, 32)
        self.transition2 = Transition(128+32*12, 256)

        self.block3 = DenseBlock(24, 256, 4, 32)
        self.transition3 = Transition(256+32*24, 512)

        self.block4 = DenseBlock(16, 512, 4, 32)

        self.bn2 = nn.BatchNorm2d(512+32*16)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool2 = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512+32*16, 10)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.pool1(out)
        out = self.block1(out)
        out = self.transition1(out)
        out = self.block2(out)
        out = self.transition2(out)
        out = self.block3(out)
        out = self.transition3(out)
        out = self.block4(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.pool2(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out

class DenseNet_half(nn.Module):
    def __init__(self):
        super(DenseNet_half, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 7, 2, 3)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(3, 2, 1)

        self.block1 = DenseBlock(6, 32, 4, 16)
        self.transition1 = Transition(32+16*6, 64)

        self.block2 = DenseBlock(12, 64, 4, 16)
        self.transition2 = Transition(64+16*12, 128)

        self.block3 = DenseBlock(24, 128, 4, 16)
        self.transition3 = Transition(128+16*24, 256)

        self.block4 = DenseBlock(16, 256, 4, 16)

        self.bn2 = nn.BatchNorm2d(256+16*16)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool2 = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256+16*16, 10)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.pool1(out)
        out = self.block1(out)
        out = self.transition1(out)
        out = self.block2(out)
        out = self.transition2(out)
        out = self.block3(out)
        out = self.transition3(out)
        out = self.block4(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.pool2(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out
