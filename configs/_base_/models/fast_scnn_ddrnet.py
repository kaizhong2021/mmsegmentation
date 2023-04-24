# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True, momentum=0.01)
data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255)
model = dict(
    type='EncoderDecoder',
    data_preprocessor=data_preprocessor,
    backbone=dict(
        type='DDRNet',
        in_channels=3,
        channels=32,
        ppm_channels=128,
        norm_cfg=norm_cfg,
        align_corners=False,
        init_cfg=dict(
            type='Pretrained',
            checkpoint='pretrained/ddrnet23s_in1k_mmseg.pth')),
    decode_head=dict(
        type='DepthwiseSeparableFCNHead',
        in_channels=64,
        channels=4,
        concat_input=False,
        num_classes=19,
        in_index=0,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1)),
    auxiliary_head=[
        dict(
            type='FCNHead',
            in_channels=128,
            channels=4,
            num_convs=1,
            num_classes=19,
            in_index=1,
            norm_cfg=norm_cfg,
            concat_input=False,
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=True, loss_weight=0.4)),
        dict(
            type='FCNHead',
            in_channels=128,
            channels=4,
            num_convs=1,
            num_classes=19,
            in_index=1,
            norm_cfg=norm_cfg,
            concat_input=False,
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=True, loss_weight=0.4)),
    ],
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
