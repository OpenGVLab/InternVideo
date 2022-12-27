# model settings
evidence_loss = dict(type='EvidenceLoss',
                      num_classes=101,
                      evidence='exp',
                      loss_type='log',
                      with_avuloss=True,
                      annealing_method='exp')
model = dict(
    type='Recognizer3D',
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
            downsample_scales=((1, 1, 1), (1, 1, 1))),
        aux_head_cfg=dict(out_channels=101, loss_weight=0.5)),
    cls_head=dict(
        type='TPNHead',
        loss_cls=evidence_loss,
        num_classes=101,
        in_channels=2048,
        spatial_type='avg',
        consensus=dict(type='AvgConsensus', dim=1),
        dropout_ratio=0.5,
        init_std=0.01))
train_cfg = None
test_cfg = dict(average_clips='prob')
dataset_type = 'VideoDataset'
data_root = 'data/ucf101/videos'
data_root_val = 'data/ucf101/videos'
ann_file_train = 'data/ucf101/ucf101_train_split_1_videos.txt'
ann_file_val = 'data/ucf101/ucf101_val_split_1_videos.txt'
ann_file_test = 'data/ucf101/ucf101_val_split_1_videos.txt'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_bgr=False)
train_pipeline = [
    dict(type='OpenCVInit', num_threads=1),
    dict(type='SampleFrames', clip_len=8, frame_interval=8, num_clips=1),
    dict(type='OpenCVDecode'),
    dict(type='RandomResizedCrop'),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='Flip', flip_ratio=0.5),
    dict(type='ColorJitter', color_space_aug=True),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'label'])
]
val_pipeline = [
    dict(type='OpenCVInit', num_threads=1),
    dict(
        type='SampleFrames',
        clip_len=8,
        frame_interval=8,
        num_clips=1,
        test_mode=True),
    dict(type='OpenCVDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='ColorJitter', color_space_aug=True),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
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
    videos_per_gpu=8,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        ann_file=ann_file_train,
        data_prefix=data_root,
        start_index=0,
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        data_prefix=data_root_val,
        start_index=0,
        pipeline=val_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=ann_file_test,
        data_prefix=data_root_val,
        start_index=0,
        pipeline=test_pipeline))
# optimizer
optimizer = dict(
    type='SGD', lr=0.002, momentum=0.9, weight_decay=0.0001,
    nesterov=True)  # this lr is used for 8 gpus
optimizer_config = dict(grad_clip=dict(max_norm=40, norm_type=2))
# learning policy
lr_config = dict(policy='step', step=[20, 40])
total_epochs = 50
checkpoint_config = dict(interval=10)
evaluation = dict(
    interval=5, metrics=['top_k_accuracy', 'mean_class_accuracy'])
log_config = dict(
    interval=20,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook'),
    ])
annealing_runner = True
# runtime settings
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './work_dirs/tpn_slowonly/finetune_ucf101_tpn_slowonly_edlloss_avuc'  # noqa: E501
load_from = 'https://download.openmmlab.com/mmaction/recognition/tpn/tpn_imagenet_pretrained_slowonly_r50_8x8x1_150e_kinetics_rgb/tpn_imagenet_pretrained_slowonly_r50_8x8x1_150e_kinetics_rgb_20200923-52629684.pth'
resume_from = None
workflow = [('train', 1)]
