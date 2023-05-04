import torch
import torch.nn as nn
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from functools import partial
from typing import List
from torch import Tensor
import copy
from mmseg.registry import MODELS


class Partial_conv3(nn.Module):

    def __init__(self, dim, n_div, forward):
        super().__init__()
        self.dim_conv3 = dim // n_div
        self.dim_untouched = dim - self.dim_conv3
        self.partial_conv3 = nn.Conv2d(self.dim_conv3, self.dim_conv3, 3, 1, 1, bias=False)

        if forward == 'slicing':
            self.forward = self.forward_slicing
        elif forward == 'split_cat':
            self.forward = self.forward_split_cat
        else:
            raise NotImplementedError

    def forward_slicing(self, x: Tensor) -> Tensor:
        # only for inference
        x = x.clone()  # !!! Keep the original input intact for the residual connection later
        x[:, :self.dim_conv3, :, :] = self.partial_conv3(x[:, :self.dim_conv3, :, :])

        return x

    def forward_split_cat(self, x: Tensor) -> Tensor:
        # for training/inference
        x1, x2 = torch.split(x, [self.dim_conv3, self.dim_untouched], dim=1)
        x1 = self.partial_conv3(x1)
        x = torch.cat((x1, x2), 1)

        return x


class MLPBlock(nn.Module):

    def __init__(self,
                 dim,
                 n_div,
                 mlp_ratio,
                 drop_path,
                 layer_scale_init_value,
                 act_layer,
                 norm_layer,
                 pconv_fw_type
                 ):

        super().__init__()
        self.dim = dim
        self.mlp_ratio = mlp_ratio
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.n_div = n_div

        mlp_hidden_dim = int(dim * mlp_ratio)

        mlp_layer: List[nn.Module] = [
            nn.Conv2d(dim, mlp_hidden_dim, 1, bias=False),
            norm_layer(mlp_hidden_dim),
            act_layer(),
            nn.Conv2d(mlp_hidden_dim, dim, 1, bias=False)
        ]

        self.mlp = nn.Sequential(*mlp_layer)

        self.spatial_mixing = Partial_conv3(
            dim,
            n_div,
            pconv_fw_type
        )

        if layer_scale_init_value > 0:
            self.layer_scale = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            self.forward = self.forward_layer_scale
        else:
            self.forward = self.forward

    def forward(self, x: Tensor) -> Tensor:
        shortcut = x
        x = self.spatial_mixing(x)
        x = shortcut + self.drop_path(self.mlp(x))
        return x

    def forward_layer_scale(self, x: Tensor) -> Tensor:
        shortcut = x
        x = self.spatial_mixing(x)
        x = shortcut + self.drop_path(
            self.layer_scale.unsqueeze(-1).unsqueeze(-1) * self.mlp(x))
        return x


class BasicStage(nn.Module):

    def __init__(self,
                 dim,
                 depth,
                 n_div,
                 mlp_ratio,
                 drop_path,
                 layer_scale_init_value,
                 norm_layer,
                 act,
                 pconv_fw_type='slicing'
                 ):
        super().__init__()

        self.blocks = nn.ModuleList([
            MLPBlock(
                dim=dim,
                n_div=n_div,
                mlp_ratio=mlp_ratio,
                drop_path=drop_path * i / depth,
                layer_scale_init_value=layer_scale_init_value,
                act_layer=act,
                norm_layer=norm_layer,
                pconv_fw_type=pconv_fw_type
            )
            for i in range(depth)
        ])

    def forward(self, x: Tensor) -> Tensor:
        for block in self.blocks:
            x = block(x)
        return x


@MODELS.register_module()
class FasterNet(nn.Module):
    def __init__(self,
                 image_size=224,
                 in_channels=3,
                 patch_size=16,
                 num_classes=1000,
                 depth=[2, 2, 6, 2],
                 dim=[64, 128, 320, 512],
                 n_divs=[4, 4, 8, 8],
                 mlp_ratios=[4, 4, 4, 4],
                 drop_path=0.1,
                 layer_scale_init_value=0,
                 norm_layer=nn.LayerNorm,
                 act_layer=nn.GELU,
                 pconv_fw_type='slicing'
                 ):
        super().__init__()

        assert image_size % patch_size == 0, 'image size must be divisible by patch size'
        num_patches = (image_size // patch_size) ** 2
        patch_dim = in_channels * patch_size ** 2

        self.patch_embed = nn.Conv2d(
            in_channels=in_channels,
            out_channels=dim[0],
            kernel_size=patch_size,
            stride=patch_size,
            bias=False
        )
        self.patch_to_embedding = nn.Linear(patch_dim, dim[0])

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, dim[0]))
        trunc_normal_(self.pos_embed, std=0.02)

        self.stage1 = BasicStage(
            dim=dim[0],
            depth=depth[0],
            n_div=n_divs[0],
            mlp_ratio=mlp_ratios[0],
            drop_path=drop_path,
            layer_scale_init_value=layer_scale_init_value,
            norm_layer=norm_layer,
            act=act_layer,
            pconv_fw_type=pconv_fw_type
        )

        self.stage2 = BasicStage(
            dim=dim[1],
            depth=depth[1],
            n_div=n_divs[1],
            mlp_ratio=mlp_ratios[1],
            drop_path=drop_path,
            layer_scale_init_value=layer_scale_init_value,
            norm_layer=norm_layer,
            act=act_layer,
            pconv_fw_type=pconv_fw_type
        )

        self.stage3 = BasicStage(
            dim=dim[2],
            depth=depth[2],
            n_div=n_divs[2],
            mlp_ratio=mlp_ratios[2],
            drop_path=drop_path,
            layer_scale_init_value=layer_scale_init_value,
            norm_layer=norm_layer,
            act=act_layer,
            pconv_fw_type=pconv_fw_type
        )

        self.stage4 = BasicStage(
            dim=dim[3],
            depth=depth[3],
            n_div=n_divs[3],
            mlp_ratio=mlp_ratios[3],
            drop_path=drop_path,
            layer_scale_init_value=layer_scale_init_value,
            norm_layer=norm_layer,
            act=act_layer,
            pconv_fw_type=pconv_fw_type
        )

        self.norm = norm_layer(dim[-1])
        self.head = nn.Linear(dim[-1], num_classes)

    def forward(self, x):
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)
        x = self.patch_to_embedding(x)
        x = x + self.pos_embed
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)

        x = self.norm(x[:, 0])
        x = self.head(x)

        return x
