_base_ = './stdc1_4xb12-80k_cityscapes-512x1024.py'
custom_imports = dict(imports=['mmpretrain.models'], allow_failed_imports=False)
model = dict(backbone=dict(
        type='RepVGG',
        arch='D2se',
        out_indices=(3, ),
    ))
