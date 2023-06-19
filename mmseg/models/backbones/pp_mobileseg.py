import torch
import torch.nn as nn
import torch.nn.functional as F

from mmseg.models.utils import resize
from mmseg.registry import MODELS
from mmengine.model import BaseModule, ModuleList


@MODELS.register_module()
class PPMobileSeg(BaseModule):
    """
    PP-MobileSeg model in PyTorch, adapted from PaddleSeg implementation
    """

    def __init__(self,
                 backbone_cfg,
                 head_cfg,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(PPMobileSeg, self).__init__()

        self.init_weights(pretrained=pretrained)

    def init_weights(self, pretrained=None):
        super(PPMobileSeg, self).init_weights(pretrained)
        self.backbone.init_weights(pretrained)
        self.head.init_weights(pretrained)

    def forward(self, img, img_meta):
        x = self.backbone(img)
        x = self.head(x)
        x = resize(
            input=x,
            size=img_meta[0]['ori_shape'][:2],
            mode='bilinear',
            align_corners=False)
        return x


class PPMobileSegHead(nn.Module):
    """
    The head of PP-MobileSeg model in PyTorch, adapted from PaddleSeg implementation
    """

    def __init__(self,
                 in_channels,
                 channels,
                 num_classes,
                 dropout_ratio=0.1):
        super(PPMobileSegHead, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, channels, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(dropout_ratio)
        self.conv_seg = nn.Conv2d(channels, num_classes, kernel_size=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.dropout(x)
        x = self.conv_seg(x)
        return x
