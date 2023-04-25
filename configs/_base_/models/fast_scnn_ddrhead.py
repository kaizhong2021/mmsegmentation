# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True, momentum=0.01)
class_weight = [
    0.8373, 0.918, 0.866, 1.0345, 1.0166, 0.9969, 0.9754, 1.0489, 0.8786,
    1.0023, 0.9539, 0.9843, 1.1116, 0.9037, 1.0865, 1.0955, 1.0865, 1.1529,
    1.0507
]
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
        type='FastSCNN',
        downsample_dw_channels=(32, 48),
        global_in_channels=64,
        global_block_channels=(64, 96, 128),
        global_block_strides=(2, 2, 1),
        global_out_channels=128,
        higher_in_channels=64,
        lower_in_channels=128,
        fusion_out_channels=128,
        out_indices=(0,1),
        norm_cfg=norm_cfg,
        align_corners=False),
    decode_head=dict(
        type='DDRHead',
        in_channels=64 * 2,
        channels=64,
        dropout_ratio=0.,
        num_classes=19,
        align_corners=False,
        norm_cfg=norm_cfg,
        loss_decode=[
            dict(
                type='OhemCrossEntropy',
                thres=0.9,
                min_kept=131072,
                class_weight=class_weight,
                loss_weight=1.0),
            dict(
                type='OhemCrossEntropy',
                thres=0.9,
                min_kept=131072,
                class_weight=class_weight,
                loss_weight=0.4),
        ]),
    auxiliary_head=[
        dict(
            type='FCNHead',
            in_channels=64,
            channels=32,
            num_convs=1,
            num_classes=19,
            in_index=-2,
            norm_cfg=norm_cfg,
            concat_input=False,
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=True, loss_weight=0.4)),
        dict(
            type='FCNHead',
            in_channels=64,
            channels=32,
            num_convs=1,
            num_classes=19,
            in_index=-2,
            norm_cfg=norm_cfg,
            concat_input=False,
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=True, loss_weight=0.4)),
    ],
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
