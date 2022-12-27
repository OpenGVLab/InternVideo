# model settings
model = dict(
    type='Recognizer3DBNN',
    backbone=dict(
        type='ResNet3dSlowOnly',
        depth=50,
        pretrained='torchvision://resnet50',
        lateral=False,
        out_indices=(2, 3),
        conv1_kernel=(1, 7, 7),
        conv1_stride_t=1,
        pool1_stride_t=1,
        inflate=(0, 0, 1, 1),
        norm_eval=False),
    neck=dict(
        type='TPN',
        in_channels=(1024, 2048),
        out_channels=1024,
        spatial_modulation_cfg=dict(
            in_channels=(1024, 2048), out_channels=2048),
        temporal_modulation_cfg=dict(downsample_scales=(8, 8)),
        upsample_cfg=dict(scale_factor=(1, 1, 1)),
        downsample_cfg=dict(downsample_scale=(1, 1, 1)),
        level_fusion_cfg=dict(
            in_channels=(1024, 1024),
            mid_channels=(1024, 1024),
            out_channels=2048,
            downsample_scales=((1, 1, 1), (1, 1, 1)))),
    cls_head=dict(
        type='TPNBNNHead',
        num_classes=101,
        in_channels=2048,
        spatial_type='avg',
        consensus=dict(type='AvgConsensus', dim=1),
        dropout_ratio=0,  # for bnn, dropout is not necessary
        init_std=0.01))
test_cfg = dict(average_clips='prob', npass=10)
dataset_type = 'VideoDataset'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_bgr=False)
test_pipeline = [
    dict(type='OpenCVInit', num_threads=1),
    dict(
        type='SampleFrames',
        clip_len=8,
        frame_interval=8,
        num_clips=10,
        test_mode=True),
    dict(type='OpenCVDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='ThreeCrop', crop_size=256),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
data = dict(
    videos_per_gpu=2,
    workers_per_gpu=2,
    test=dict(
        type=dataset_type,
        ann_file=None,
        data_prefix=None,
        pipeline=test_pipeline))
