import os
from torchvision import transforms
from .transforms import *
from .masking_generator import TubeMaskingGenerator, RandomMaskingGenerator
from .mae import VideoMAE
from .mae_multi import VideoMAE_multi
from .kinetics import VideoClsDataset
from .kinetics_sparse import VideoClsDataset_sparse
from .anet import ANetDataset
from .ssv2 import SSVideoClsDataset, SSRawFrameClsDataset
from .hmdb import HMDBVideoClsDataset, HMDBRawFrameClsDataset


class DataAugmentationForVideoMAE(object):
    def __init__(self, args):
        self.input_mean = [0.485, 0.456, 0.406]  # IMAGENET_DEFAULT_MEAN
        self.input_std = [0.229, 0.224, 0.225]  # IMAGENET_DEFAULT_STD
        normalize = GroupNormalize(self.input_mean, self.input_std)
        self.train_augmentation = GroupMultiScaleCrop(args.input_size, [1, .875, .75, .66])
        if args.color_jitter > 0:
            self.transform = transforms.Compose([                            
                self.train_augmentation,
                GroupColorJitter(args.color_jitter),
                GroupRandomHorizontalFlip(flip=args.flip),
                Stack(roll=False),
                ToTorchFormatTensor(div=True),
                normalize,
            ])
        else:
            self.transform = transforms.Compose([                            
                self.train_augmentation,
                GroupRandomHorizontalFlip(flip=args.flip),
                Stack(roll=False),
                ToTorchFormatTensor(div=True),
                normalize,
            ])
        if args.mask_type == 'tube':
            self.masked_position_generator = TubeMaskingGenerator(
                args.window_size, args.mask_ratio
            )
        elif args.mask_type == 'random':
            self.masked_position_generator = RandomMaskingGenerator(
                args.window_size, args.mask_ratio
            )
        elif args.mask_type in 'attention':
            self.masked_position_generator = None

    def __call__(self, images):
        process_data, _ = self.transform(images)
        if self.masked_position_generator is None:
            return process_data, -1
        else:
            return process_data, self.masked_position_generator()

    def __repr__(self):
        repr = "(DataAugmentationForVideoMAE,\n"
        repr += "  transform = %s,\n" % str(self.transform)
        repr += "  Masked position generator = %s,\n" % str(self.masked_position_generator)
        repr += ")"
        return repr


def build_pretraining_dataset(args):
    transform = DataAugmentationForVideoMAE(args)
    dataset = VideoMAE(
        root=None,
        setting=args.data_path,
        prefix=args.prefix,
        split=args.split,
        video_ext='mp4',
        is_color=True,
        modality='rgb',
        num_segments=args.num_segments,
        new_length=args.num_frames,
        new_step=args.sampling_rate,
        transform=transform,
        temporal_jitter=False,
        video_loader=True,
        use_decord=args.use_decord,
        lazy_init=False,
        num_sample=args.num_sample)
    print("Data Aug = %s" % str(transform))
    return dataset


def build_multi_pretraining_dataset(args):
    origianl_flip = args.flip
    transform = DataAugmentationForVideoMAE(args)
    args.flip = False
    transform_ssv2 = DataAugmentationForVideoMAE(args)
    args.flip = origianl_flip

    dataset = VideoMAE_multi(
        root=None,
        setting=args.data_path,
        prefix=args.prefix,
        split=args.split,
        is_color=True,
        modality='rgb',
        num_segments=args.num_segments,
        new_length=args.num_frames,
        new_step=args.sampling_rate,
        transform=transform,
        transform_ssv2=transform_ssv2,
        temporal_jitter=False,
        video_loader=True,
        use_decord=args.use_decord,
        lazy_init=False,
        num_sample=args.num_sample)
    print("Data Aug = %s" % str(transform))
    print("Data Aug for SSV2 = %s" % str(transform_ssv2))
    return dataset


def build_dataset(is_train, test_mode, args):
    print(f'Use Dataset: {args.data_set}')
    if args.data_set in [
            'Kinetics',
            'Kinetics_sparse',
            'mitv1_sparse'
        ]:
        mode = None
        anno_path = None
        if is_train is True:
            mode = 'train'
            anno_path = os.path.join(args.data_path, 'train.csv')
        elif test_mode is True:
            mode = 'test'
            anno_path = os.path.join(args.data_path, 'test.csv') 
        else:  
            mode = 'validation'
            anno_path = os.path.join(args.data_path, 'val.csv') 

        if 'sparse' in args.data_set:
            func = VideoClsDataset_sparse
        else:
            func = VideoClsDataset

        dataset = func(
            anno_path=anno_path,
            prefix=args.prefix,
            split=args.split,
            mode=mode,
            clip_len=args.num_frames,
            frame_sample_rate=args.sampling_rate,
            num_segment=1,
            test_num_segment=args.test_num_segment,
            test_num_crop=args.test_num_crop,
            num_crop=1 if not test_mode else 3,
            keep_aspect_ratio=True,
            crop_size=args.input_size,
            short_side_size=args.short_side_size,
            new_height=256,
            new_width=320,
            args=args)
        
        nb_classes = args.nb_classes
    
    elif args.data_set == 'SSV2':
        mode = None
        anno_path = None
        if is_train is True:
            mode = 'train'
            anno_path = os.path.join(args.data_path, 'train.csv')
        elif test_mode is True:
            mode = 'test'
            anno_path = os.path.join(args.data_path, 'test.csv') 
        else:  
            mode = 'validation'
            anno_path = os.path.join(args.data_path, 'val.csv') 

        if args.use_decord:
            func = SSVideoClsDataset
        else:
            func = SSRawFrameClsDataset

        dataset = func(
            anno_path=anno_path,
            prefix=args.prefix,
            split=args.split,
            mode=mode,
            clip_len=1,
            num_segment=args.num_frames,
            test_num_segment=args.test_num_segment,
            test_num_crop=args.test_num_crop,
            num_crop=1 if not test_mode else 3,
            keep_aspect_ratio=True,
            crop_size=args.input_size,
            short_side_size=args.short_side_size,
            new_height=256,
            new_width=320,
            filename_tmpl=args.filename_tmpl,
            args=args)
        nb_classes = 174

    elif args.data_set == 'UCF101':
        mode = None
        anno_path = None
        if is_train is True:
            mode = 'train'
            anno_path = os.path.join(args.data_path, 'train.csv')
        elif test_mode is True:
            mode = 'test'
            anno_path = os.path.join(args.data_path, 'test.csv') 
        else:  
            mode = 'validation'
            anno_path = os.path.join(args.data_path, 'val.csv') 

        dataset = VideoClsDataset(
            anno_path=anno_path,
            prefix=args.prefix,
            split=args.split,
            mode=mode,
            clip_len=args.num_frames,
            frame_sample_rate=args.sampling_rate,
            num_segment=1,
            test_num_segment=args.test_num_segment,
            test_num_crop=args.test_num_crop,
            num_crop=1 if not test_mode else 3,
            keep_aspect_ratio=True,
            crop_size=args.input_size,
            short_side_size=args.short_side_size,
            new_height=256,
            new_width=320,
            args=args)
        nb_classes = 101
    
    elif args.data_set == 'HMDB51':
        mode = None
        anno_path = None
        if is_train is True:
            mode = 'train'
            anno_path = os.path.join(args.data_path, 'train.csv')
        elif test_mode is True:
            mode = 'test'
            anno_path = os.path.join(args.data_path, 'test.csv') 
        else:  
            mode = 'validation'
            anno_path = os.path.join(args.data_path, 'val.csv') 

        if args.use_decord:
            func = HMDBVideoClsDataset
        else:
            func = HMDBRawFrameClsDataset

        dataset = func(
            anno_path=anno_path,
            prefix=args.prefix,
            split=args.split,
            mode=mode,
            clip_len=1,
            num_segment=args.num_frames,
            test_num_segment=args.test_num_segment,
            test_num_crop=args.test_num_crop,
            num_crop=1 if not test_mode else 3,
            keep_aspect_ratio=True,
            crop_size=args.input_size,
            short_side_size=args.short_side_size,
            new_height=256,
            new_width=320,
            filename_tmpl=args.filename_tmpl,
            args=args)
        nb_classes = 51

    elif args.data_set in [
        'ANet', 
        'HACS',
        'ANet_interval', 
        'HACS_interval'
    ]:
        mode = None
        anno_path = None
        if is_train is True:
            mode = 'train'
            anno_path = os.path.join(args.data_path, 'train.csv')
        elif test_mode is True:
            mode = 'test'
            anno_path = os.path.join(args.data_path, 'test.csv') 
        else:  
            mode = 'validation'
            anno_path = os.path.join(args.data_path, 'val.csv') 

        if 'interval' in args.data_set:
            func = ANetDataset
        else:
            func = VideoClsDataset_sparse

        dataset = func(
            anno_path=anno_path,
            prefix=args.prefix,
            split=args.split,
            mode=mode,
            clip_len=args.num_frames,
            frame_sample_rate=args.sampling_rate,
            num_segment=1,
            test_num_segment=args.test_num_segment,
            test_num_crop=args.test_num_crop,
            num_crop=1 if not test_mode else 3,
            keep_aspect_ratio=True,
            crop_size=args.input_size,
            short_side_size=args.short_side_size,
            new_height=256,
            new_width=320,
            args=args)
        nb_classes = args.nb_classes

    else:
        print(f'Wrong: {args.data_set}')
        raise NotImplementedError()
    assert nb_classes == args.nb_classes
    print("Number of the class = %d" % args.nb_classes)

    return dataset, nb_classes
