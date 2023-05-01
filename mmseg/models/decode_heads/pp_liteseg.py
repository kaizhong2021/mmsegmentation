import torch.nn as nn
# from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from mmseg.registry import MODELS


@MODELS.register_module()
class PP_LiteSeg_DecodeHead(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(PP_LiteSeg_DecodeHead, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(in_channels // 4)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(in_channels // 4, num_classes, kernel_size=1, stride=1)

        self.upsample = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)

        x = self.upsample(x)

        return x
