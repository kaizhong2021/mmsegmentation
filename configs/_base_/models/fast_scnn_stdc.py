_base_ = 'fast_scnn.py'
model = dict(
    type='EncoderDecoder',
    data_preprocessor=data_preprocessor,
    backbone=dict(
        _delete_=True,
        type = 'STDCContextPathNet', 
    )
)