_base_ = 'fast_scnn_8xb4-160k_cityscapes-512x1024_stdc.py'
model = dict(backbone=dict(backbone_cfg=dict(stdc_type='STDCNet2')))
