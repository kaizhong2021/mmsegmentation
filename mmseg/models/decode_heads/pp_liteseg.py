import torch.nn as nn
import torch.nn.functional as F
from mmseg.models.utils import resize
from mmcv.cnn import ConvModule, Scale
from mmseg.registry import MODELS
from .decode_head import BaseDecodeHead


@MODELS.register_module()
class PP_Lite_DecodeHead(BaseDecodeHead):
    def __init__(self, feature_strides, **kwargs):
        super(PP_Lite_DecodeHead, self).__init__(**kwargs)
        # Define the pyramid pooling module
        self.pyramid_pool = nn.ModuleList()
        for scale in feature_strides:
            self.pyramid_pool.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(output_size=(scale, scale)),
                nn.Conv2d(self.in_channels, self.channels, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(self.channels),
                nn.ReLU(inplace=True)
            ))
        # Define the convolutional layers for each scale
        self.conv_cls = nn.ModuleList()
        for i in range(len(feature_strides)):
            self.conv_cls.append(nn.Conv2d(self.channels, self.num_classes, kernel_size=1, stride=1))

    def forward(self, inputs):
        x = self._transform_inputs(inputs)
        # Apply pyramid pooling at different scales
        pyramid = []
        for pool in self.pyramid_pool:
            pyramid.append(pool(x))
        # Concatenate the feature maps and apply the decode head
        feature_add_all = pyramid[0]
        for idx, feature in enumerate(pyramid[:-1]):
            feature_add_all = F.interpolate(feature_add_all, scale_factor=2, mode='bilinear', align_corners=self.align_corners)
            if feature_add_all.shape != feature.shape:
                feature_add_all = F.interpolate(feature_add_all, size=feature.shape[-2:], mode='bilinear',
                                                align_corners=True)

            feature_add_all = feature_add_all + feature
        seg_logits = self.conv_cls[0](feature_add_all)
        for idx, x in enumerate(pyramid):
            if idx == 0:
                continue
            if seg_logits.shape != self.conv_cls[idx](x).shape:
                seg_logits = F.interpolate(seg_logits,size=self.conv_cls[idx](x).shape[-2:],mode='bilinear',align_corners=True)
            seg_logits += self.conv_cls[idx](x)

        return seg_logits
