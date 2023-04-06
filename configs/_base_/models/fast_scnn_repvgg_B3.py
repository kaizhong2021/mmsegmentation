custom_imports = dict(imports='mmcls.models', allow_failed_imports=False)
norm_cfg = dict(type='SyncBN', requires_grad=True, momentum=0.01)
data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255,
    size=(512, 1024))
model = dict(
    type='EncoderDecoder',
    data_preprocessor=dict(
        type='SegDataPreProcessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_val=0,
        seg_pad_val=255,
        size=(512, 1024)),
    backbone=dict(type='mmcls.RepVGG', arch='B3', out_indices=(1, 2, 3)),
    decode_head=dict(
        type='DepthwiseSeparableFCNHead',
        in_channels=2560,
        channels=192,
        concat_input=False,
        num_classes=19,
        in_index=-1,
        norm_cfg=dict(type='SyncBN', requires_grad=True, momentum=0.01),
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1)),
    auxiliary_head=[
        dict(
            type='FCNHead',
            in_channels=768,
            channels=4,
            num_convs=1,
            num_classes=19,
            in_index=-2,
            norm_cfg=dict(type='SyncBN', requires_grad=True, momentum=0.01),
            concat_input=False,
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=True, loss_weight=0.4)),
        dict(
            type='FCNHead',
            in_channels=384,
            channels=4,
            num_convs=1,
            num_classes=19,
            in_index=-3,
            norm_cfg=dict(type='SyncBN', requires_grad=True, momentum=0.01),
            concat_input=False,
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=True, loss_weight=0.4))
    ],
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
