# input: (C, T, H, W) output: (C, T, H, W)
def VideoTransform(mode='train', crop_size=224, backend='v100'):
    if backend == 'a100':
        print("initalize data augmentation for a100 gpus")
        import CoTrain.transforms.video.video_transform as video_transform
        from torchvision import transforms
        # https://github.com/FingerRec/BE/blob/main/src/Contrastive/augment/video_transformations/volume_transforms.py

        input_mean = [0.48145466, 0.4578275, 0.40821073]
        input_std = [0.26862954, 0.26130258, 0.27577711]
        scale_size = crop_size * 256 // 224
        if mode == 'train':
            global_transforms = transforms.Compose([
                video_transform.TensorToNumpy(),
                # video_transform.Resize(int(crop_size * 1.2)),  # 256/224 = 1.14
                video_transform.Resize(scale_size),
                video_transform.RandomCrop(crop_size),
                # video_transform.ColorJitter(0.5, 0.5, 0.25, 0.5),  # color operation perimitted, damage attribute
                video_transform.ClipToTensor(channel_nb=3),
                # video_transform.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                video_transform.Normalize(mean=input_mean, std=input_std)
            ])
            local_transforms = transforms.Compose([
                video_transform.TensorToNumpy(),
                video_transform.Resize(crop_size),  # 256/224 = 1.14
                video_transform.RandomCrop(96),
                # video_transform.ColorJitter(0.5, 0.5, 0.25, 0.5),  # color operation perimitted, damage attribute
                video_transform.ClipToTensor(channel_nb=3),
                # video_transform.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                video_transform.Normalize(mean=input_mean, std=input_std)
            ])
        else:
            global_transforms = transforms.Compose([
                video_transform.TensorToNumpy(),
                # video_transform.Resize(int(crop_size * 1.2)),  # 256
                video_transform.Resize(scale_size),
                video_transform.CenterCrop(crop_size),  # 224
                video_transform.ClipToTensor(channel_nb=3),
                # video_transform.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                video_transform.Normalize(mean=input_mean, std=input_std)
            ])
            local_transforms = transforms.Compose([
                video_transform.TensorToNumpy(),
                video_transform.Resize(crop_size),  # 256
                video_transform.CenterCrop(96),  # 224
                video_transform.ClipToTensor(channel_nb=3),
                # video_transform.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                video_transform.Normalize(mean=input_mean, std=input_std)
            ])
        return [global_transforms, local_transforms]
    else:
        # for pytorch > 1.9.0, V100
        import pytorchvideo.transforms as video_transforms
        # https://pytorchvideo.readthedocs.io/en/latest/api/transforms/transforms.html
        global_transform = video_transforms.create_video_transform(mode=mode, min_size=int(crop_size*1.2),
                                                       max_size=int(crop_size*1.5),
                                                       crop_size=crop_size,
                                                       aug_type='randaug',  # randaug/augmix
                                                       num_samples=None)  # not use temporal sub sampling
        local_transform = video_transforms.create_video_transform(mode=mode, min_size=crop_size,
                                                       max_size=int(crop_size*1.5),
                                                       crop_size=96,
                                                       aug_type='randaug',  # randaug/augmix
                                                       num_samples=None)  # not use temporal sub sampling
        return [global_transform, local_transform]


def video_aug(videos, video_transform, byte=False):
    if byte:
        videos = videos.permute(1, 0, 2, 3).byte()  # tchw -> cthw
    else:
        videos = videos.permute(1, 0, 2, 3)
    # normal
    # videos_tensor = [video_transform(videos).permute(1, 0, 2, 3)]  # -> tchw
    # dino
    global_videos_tensor = []
    global_transform, local_transform = video_transform
    # print(videos.type())
    # 2 GLOBAL views
    for i in range(1):
        global_videos_tensor.append(global_transform(videos).permute(1, 0, 2, 3))
    # 3 LOCAL VIEWS
    # local_videos_tensor = []
    # for i in range(0):
    #     local_videos_tensor.append(local_transform(videos).permute(1, 0, 2, 3))
    return global_videos_tensor
    # return [global_videos_tensor, local_videos_tensor]


# # dino
# def video_aug(videos, video_transform, byte=False):
#     if byte:
#         videos = videos.permute(1, 0, 2, 3)  # tchw -> cthw
#     else:
#         videos = videos.permute(1, 0, 2, 3).byte()
#     # two global view
#     videos_tensor = [video_transform(videos).permute(1, 0, 2, 3), video_transform(videos).permute(1, 0, 2, 3)]  # -> tchw
#     # local views
#     return videos_tensor