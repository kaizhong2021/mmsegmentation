# decode_head.py
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from mmseg.registry import MODELS

@MODELS.register_module()
class PPMobileSegHead(BaseDecodeHead):
    """
    The PP_MobileSeg head implementation based on PyTorch.
    """
    def __init__(self,
                 num_classes,
                 in_channels,
                 channels,
                 use_dw=True,
                 dropout_ratio=0.1):
        super().__init__(
            in_channels,
            channels,
            num_classes=num_classes,
        )
        self.channels = channels
        self.last_channels = in_channels
        self.conv1x1 = nn.Conv2d(
            in_channels=self.last_channels,
            out_channels=self.last_channels,
            kernel_size=1,
            stride=1,
            groups=self.last_channels if use_dw else 1)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(dropout_ratio)
        self.conv_seg = nn.Conv2d(
            self.last_channels, num_classes, kernel_size=1)

    def forward(self, x):
        x = self.conv1x1(x[0])
        x = self.relu(x)
        x = self.dropout(x)
        x = self.conv_seg(x)
        return x
