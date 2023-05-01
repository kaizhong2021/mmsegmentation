import torch
import torch.nn as nn
from mmseg.registry import MODELS

@MODELS.register_module()
class PP_Lite_DecodeHead(nn.Module):
    def __init__(self, in_channels, num_classes, align_corners=True):
        super().__init__()
        self.align_corners = align_corners
        self.conv1 = nn.Conv2d(in_channels, 256, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(256)
        self.relu1 = nn.ReLU(inplace=True)
        self.dropout1 = nn.Dropout2d(0.1)

        self.conv2 = nn.Conv2d(256, num_classes, kernel_size=1, stride=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.conv2(x)

        if self.align_corners:
            x = nn.functional.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        else:
            x = nn.functional.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        return x
