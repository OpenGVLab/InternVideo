model = dict(
    type='Recognizer3D',
    backbone=dict(
        type='ResNet3dSlowFast',
        pretrained=None,
        resample_rate=4,  # tau
        speed_ratio=4,  # alpha
        channel_ratio=8,  # beta_inv
        slow_pathway=dict(
            type='resnet3d',
            depth=50,
            pretrained=None,
            lateral=True,
            fusion_kernel=7,
            conv1_kernel=(1, 7, 7),
            dilations=(1, 1, 1, 1),
            conv1_stride_t=1,
            pool1_stride_t=1,
            inflate=(0, 0, 1, 1),
            norm_eval=False),
        fast_pathway=dict(
            type='resnet3d',
            depth=50,
            pretrained=None,
            lateral=False,
            base_channels=8,
            conv1_kernel=(5, 7, 7),
            conv1_stride_t=1,
            pool1_stride_t=1,
            norm_eval=False)),
    cls_head=dict(
        type='SlowFastHead',
        loss_cls=dict(type='EvidenceLoss',
                      num_classes=101,
                      evidence='exp',
                      loss_type='log',
                      annealing_method='exp'),
        in_channels=2304,  # 2048+256
        num_classes=101,
        spatial_type='avg',
        dropout_ratio=0.5))
evidence='exp'  # only used for EDL
test_cfg = dict(average_clips='score')
dataset_type = 'VideoDataset'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_bgr=False)
test_pipeline = [
    # dict(type='OpenCVInit', num_threads=1),
    dict(type='DecordInit'),
    dict(
        type='SampleFrames',
        clip_len=32,
        frame_interval=2,
        num_clips=10,
        test_mode=True),
    # dict(type='OpenCVDecode'),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='ThreeCrop', crop_size=256),
    dict(type='Flip', flip_ratio=0),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
data = dict(
    videos_per_gpu=8,
    workers_per_gpu=2,
    test=dict(
        type=dataset_type,
        ann_file=None,
        data_prefix=None,
        start_index=0,
        pipeline=test_pipeline))
dist_params = dict(backend='nccl')