_base_ = 'fast_scnn.py'
norm_cfg = dict(type='BN', requires_grad=True)
model = dict(
    type='EncoderDecoder',
    backbone=dict(
        _delete_=True,
        type = 'STDCContextPathNet',
        backbone_cfg=dict(
            type='STDCNet',
            stdc_type='STDCNet1',
            in_channels=3,
            channels=(32, 64, 256, 512, 1024),
            bottleneck_type='cat',
            num_convs=4,
            norm_cfg=norm_cfg,
            act_cfg=dict(type='ReLU'),
            with_final_conv=True),
        last_in_channels=(1024, 512),
        out_channels=128,
        ffm_cfg=dict(in_channels=384, out_channels=256, scale_factor=4),
    )
)

