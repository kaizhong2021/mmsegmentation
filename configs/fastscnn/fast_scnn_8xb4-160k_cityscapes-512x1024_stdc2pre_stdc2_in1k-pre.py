_base_ = 'fast_scnn_8xb4-160k_cityscapes-512x1024_stdc2.py'
checkpoint = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/stdc/stdc2_20220308-7dbd9127.pth'  # noqa
model = dict(
    backbone=dict(
        backbone_cfg=dict(
            init_cfg=dict(type='Pretrained', checkpoint=checkpoint))))
