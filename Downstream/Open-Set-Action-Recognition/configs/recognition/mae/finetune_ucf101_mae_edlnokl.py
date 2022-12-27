# model settings
evidence_loss = dict(type='EvidenceLoss',
                      num_classes=101,
                      evidence='exp',
                      loss_type='log',
                      with_kldiv=False,
                      with_avuloss=False,
                      annealing_method='exp')

# mae huge ------------
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
        pretrained='work_dirs/mae/finetune_ucf101_mae_dnn/huangbingkun/model/vit_h_hybridv2_pt_1200e_k700_ft_rep_2.pth'
    ),
    cls_head=dict(
        type='BaseClsHead',
        loss_cls=evidence_loss,
        in_channels=1280,
        num_classes=101,
        dropout_ratio=0.5,
    ))


# model training and testing settings
train_cfg = None
test_cfg = dict(average_clips='evidence', evidence_type='exp')
# dataset settings
dataset_type = 'VideoDataset'
data_root = 'data/ucf101/videos'
data_root_val = 'data/ucf101/videos'
ann_file_train = 'data/ucf101/ucf101_train_split_1_videos.txt'
ann_file_val = 'data/ucf101/ucf101_val_split_1_videos.txt'
ann_file_test = 'data/ucf101/ucf101_val_split_1_videos.txt'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_bgr=False)
train_pipeline = [
    # dict(type='OpenCVInit', num_threads=1),
    dict(type='DecordInit'),
    dict(type='DenseSampleFrames', clip_len=32, frame_interval=2, num_clips=1),
    # dict(type='SampleFrames', clip_len=1, frame_interval=1, num_clips=32),
    # dict(type='SampleFrames', clip_len=32, frame_interval=2, num_clips=1),
    # dict(type='OpenCVDecode'),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(
        type='MultiScaleCrop',
        input_size=224,
        scales=(1, 0.8),
        random_crop=False,
        max_wh_scale_gap=0),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='Flip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'label'])
]
val_pipeline = [
    # dict(type='OpenCVInit', num_threads=1),
    dict(type='DecordInit'),
    dict(
        type='DenseSampleFrames',
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
    # dict(
    #     type='SampleFrames',
    #     clip_len=32,
    #     frame_interval=2,
    #     num_clips=1,
    #     test_mode=True),
    # dict(type='OpenCVDecode'),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='Flip', flip_ratio=0),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
test_pipeline = [
    # dict(type='OpenCVInit', num_threads=1),
    dict(type='DecordInit'),
    dict(
        type='DenseSampleFrames',
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
    # dict(
    #     type='SampleFrames',
    #     clip_len=32,
    #     frame_interval=2,
    #     num_clips=1,
    #     test_mode=True),
    # dict(type='OpenCVDecode'),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 224)),
    dict(type='ThreeCrop', crop_size=224),
    # dict(type='CenterCrop', crop_size=224),
    dict(type='Flip', flip_ratio=0),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
data = dict(
    videos_per_gpu=1,  # set to 2 for evaluation on GPU with 24GB 
    workers_per_gpu=4,  # set to 2 for evaluation on GPU with 24GB 
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
        ann_file=ann_file_val,
        start_index=0,
        data_prefix=data_root_val,
        pipeline=test_pipeline))
# optimizer
optimizer = dict(
    type='SGD', lr=0.001, momentum=0.9,   # change from 0.01 to 0.001
    weight_decay=0.0001, nesterov=True) 

optimizer_config = dict(grad_clip=dict(max_norm=40, norm_type=2))
# learning policy
lr_config = dict(policy='step', step=[20, 40])  # change from [40,80] to [20,40]
total_epochs = 50 # change from 100 to 50
checkpoint_config = dict(interval=10)
evaluation = dict(
    interval=60, metrics=['top_k_accuracy', 'mean_class_accuracy'])
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
work_dir = './work_dirs/finetune_ucf101_mae_edlnokl/'
# load_from = 'https://download.openmmlab.com/mmaction/recognition/i3d/i3d_r50_dense_256p_32x2x1_100e_kinetics400_rgb/i3d_r50_dense_256p_32x2x1_100e_kinetics400_rgb_20200725-24eb54cc.pth'  # model path can be found in model zoo
load_from = None
resume_from = None
workflow = [('train', 1)]
