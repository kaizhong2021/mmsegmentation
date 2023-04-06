_base_ = 'fast_scnn_copy.py'
custom_imports=dict(imports='mmcls.models', allow_failed_imports=False)
model = dict(
    backbone=dict(
        _delete_=True,
        type='mmcls.RepVGG',
        arch='A0',
        out_indices=(1,2,3),
    )
)


