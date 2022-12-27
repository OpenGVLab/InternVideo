# model settings
evidence_loss = dict(type='EvidenceLoss',
                      num_classes=101,
                      evidence='exp',
                      loss_type='log',
                      with_kldiv=False,
                      with_avuloss=False,
                      annealing_method='exp')

# mae huge
model = dict(
    type='Recognizer3D',
    backbone=dict(
        type='VisionTransformer3D',
        patch_size=16,
        embed_dim=1280,
        depth=32,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
        num_classes=0,
        # pretrained='work_dirs/mae/finetune_ucf101_mae_dnn/huangbingkun/model/vit_h_hybridv2_pt_1200e_k700_ft_rep_2.pth'
    ),
    cls_head=dict(
        type='BaseClsHead',
        loss_cls=evidence_loss,
        in_channels=1280,
        num_classes=101,
        dropout_ratio=0.5,
    )) 

# model training and testing settings
evidence='exp'  # only used for EDL
test_cfg = dict(average_clips='score')
# dataset settings
dataset_type = 'VideoDataset'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_bgr=False)
test_pipeline = [
    dict(type='DecordInit'),
    dict(
        type='SampleFrames',
        clip_len=32,
        frame_interval=2,
        num_clips=1,
        test_mode=True),
    # dict(
    #     type='SampleFrames',
    #     clip_len=1,
    #     frame_interval=1,
    #     num_clips=32,
    #     test_mode=True),
    # dict(type='OpenCVDecode'),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 224)),
    dict(type='ThreeCrop', crop_size=224),
    # dict(type='CenterCrop', crop_size=224),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
data = dict(
    videos_per_gpu=4,
    workers_per_gpu=4,
    test=dict(
        type=dataset_type,
        ann_file=None,
        data_prefix=None,
        pipeline=test_pipeline))
dist_params = dict(backend='nccl')