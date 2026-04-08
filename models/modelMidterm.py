import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.use_shortcut = (stride != 1 or in_channels != out_channels)

        if self.use_shortcut:
            self.shortcut_conv = nn.Conv2d(in_channels, out_channels, 1, stride, bias=False)
            self.shortcut_bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):

        if self.use_shortcut:
            res = self.shortcut_bn(self.shortcut_conv(x))
        else:
            res = x

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        out = out + res
        out = F.relu(out)

        return out

class M1(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        self.conv1_s1 = nn.Conv2d(3, 32, 3, 1, 1, bias=False)
        self.conv1_s2 = nn.Conv2d(3, 32, 3, 2, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)

        self.stage1 = nn.Sequential(
            ResidualBlock(32, 64, 1),
            ResidualBlock(64, 64, 1)
        )

        self.stage2 = nn.Sequential(
            ResidualBlock(64, 128, 2),
            ResidualBlock(128, 128, 1)
        )

        self.stage3 = nn.Sequential(
            ResidualBlock(128, 256, 2),
            ResidualBlock(256, 256, 1)
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):

        if x.shape[-1] == 224:
            x = self.conv1_s2(x)
        else:
            x = self.conv1_s1(x)

        x = F.relu(self.bn1(x))

        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)

        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x