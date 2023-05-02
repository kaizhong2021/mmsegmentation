import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from mmseg.registry import MODELS
from .decode_head import BaseDecodeHead
_interpolate = partial(F.interpolate, mode="bilinear", align_corners=True)

@MODELS.register_module()
class LPSNet(BaseDecodeHead):
    def __init__(self, depths, channel, scale_ratios, num_classes, in_channels=3, pretrained=None):
        super().__init__(
            in_channels,
            channel,
            num_classes=num_classes
        )

        self.depths = depths
        self.channel = channel
        self.scale_ratios = list(filter(lambda x: x > 0, scale_ratios))
        self.num_classes = num_classes
        self.in_channels = in_channels

        self.num_paths = len(self.scale_ratios)
        self.num_blocks = len(depths)

        self.nets = nn.ModuleList(
            [self._build_path() for _ in range(self.num_paths)])

        self.head = nn.Conv2d(
            channel * self.num_paths, num_classes, 1, bias=True)

        self._init_weight(pretrained)

    def _build_path(self):
        path = []
        c_in = self.in_channels
        for b, d in enumerate(self.depths):
            blocks = []
            for i in range(d):
                blocks.append(
                    nn.Sequential(
                        nn.Conv2d(
                            in_channels=self.in_channels,  # change in_channels here
                            out_channels=self.channel,
                            kernel_size=3,
                            padding=1,
                            stride=2
                            if (i == 0 and b != self.num_blocks - 1) else 1,
                            bias=False, ),
                        nn.BatchNorm2d(self.channel),
                        nn.ReLU(inplace=True)
                    )
                )
            path.append(nn.Sequential(*blocks))
        return nn.ModuleList(path)

    def _init_weight(self, pretrained):
        if pretrained is not None:
            state_dict = torch.load(pretrained, map_location=torch.device('cpu'))
            self.load_state_dict(state_dict)

    def _preprocess_input(self, x):
        if isinstance(x, tuple):
            x = tuple(torch.tensor(t) if not torch.is_tensor(t) else t for t in x)
        elif isinstance(x, list):
            x = [torch.tensor(t) if not torch.is_tensor(t) else t for t in x]
        else:
            x = torch.tensor(x) if not torch.is_tensor(x) else x

        if isinstance(x, tuple) or isinstance(x, list):
            # Reshape the tensor to have more than one element
            x = [t.view(1, self.in_channels, -1, -1) if t.dim() == 3 else t for t in x]  # change here

            # Get the shape of the first tensor in the list
            h, w = x[0].shape[-2:]
        else:
            if x.dim() == 3:  # change here
                x = x.view(1, self.in_channels, -1, -1)  # change here

            h, w = x.shape[-2:]

        return [
            _interpolate(t, (int(r * h), int(r * w))) for r in self.scale_ratios for t in x
        ]

    def forward(self, x, interact_begin_idx=2):
        input_size = x[0].size()[-2:]
        inputs = self._preprocess_input(x)
        inputs = [torch.cat([t] * self.in_channels, dim=1) for t in inputs]  # 将inputs的in_channels维度都扩展到相同的大小
        feats = []
        for path, x in zip(self.nets, inputs):
            inp = x
            for idx in range(interact_begin_idx + 1):
                inp = path[idx](inp)
            feats.append(inp)

        for idx in range(interact_begin_idx + 1, self.num_blocks):
            feats = _multipath_interaction(feats)
            feats = [path[idx](x) for path, x in zip(self.nets, feats)]

        size = feats[0].size()[-2:]
        feats = [_interpolate(x, size=size) for x in feats]

        out = self.head(torch.cat(feats, 1))

        return [_interpolate(out, size=input_size)]

def _multipath_interaction(feats):
    length = len(feats)
    if length == 1:
        return feats[0]
    sizes = [x.size()[-2:] for x in feats]
    outs = []
    looper = list(range(length))
    for i, s in enumerate(looper):
        out = feats[i]
        for j in range(length):
            if j != i:
                out = _interpolate(out, sizes[j])
                out = out + feats[j]
        outs.append(out)
    return outs