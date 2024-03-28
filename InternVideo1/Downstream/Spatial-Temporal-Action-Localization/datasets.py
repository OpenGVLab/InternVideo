import os
from torchvision import transforms
from transforms import *
from masking_generator import TubeMaskingGenerator
from kinetics import VideoClsDataset, VideoMAE

from data.ava import AVAVideoDataset,KineticsDataset,AKDataset
from data.transforms import TransformsCfg
import alphaction.config.paths_catalog as paths_catalog
from alphaction.dataset.collate_batch import batch_different_videos


class DataAugmentationForVideoMAE(object):
    def __init__(self, args):
        self.input_mean = [0.485, 0.456, 0.406] # IMAGENET_DEFAULT_MEAN
        self.input_std = [0.229, 0.224, 0.225] # IMAGENET_DEFAULT_STD
        normalize = GroupNormalize(self.input_mean, self.input_std)
        self.train_augmentation = GroupMultiScaleCrop(args.input_size, [1, .875, .75, .66])
        self.transform = transforms.Compose([                            
            self.train_augmentation,
            Stack(roll=False),
            ToTorchFormatTensor(div=True),
            normalize,
        ])
        if args.mask_type == 'tube':
            self.masked_position_generator = TubeMaskingGenerator(
                args.window_size, args.mask_ratio
            )

    def __call__(self, images):
        process_data , _ = self.transform(images)
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
        video_ext='mp4',
        is_color=True,
        modality='rgb',
        new_length=args.num_frames,
        new_step=args.sampling_rate,
        transform=transform,
        temporal_jitter=False,
        video_loader=True,
        use_decord=True,
        lazy_init=False)
    print("Data Aug = %s" % str(transform))
    return dataset


def build_dataset(is_train, test_mode, args):
    if args.data_set == 'Kinetics-400':
        mode = None
        anno_path = None
        if is_train == True:
            mode = 'train'
            anno_path = os.path.join(args.data_path, 'train.csv')
        elif test_mode == True:
            mode = 'test'
            anno_path = os.path.join(args.data_path, 'val.csv') 
        else:  
            mode = 'validation'
            anno_path = os.path.join(args.data_path, 'test.csv') 

        dataset = VideoClsDataset(
            anno_path=anno_path,
            data_path='/',
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
        nb_classes = 400
    
    elif args.data_set == 'SSV2':
        mode = None
        anno_path = None
        if is_train == True:
            mode = 'train'
            anno_path = os.path.join(args.data_path, 'train.csv')
        elif test_mode == True:
            mode = 'test'
            anno_path = os.path.join(args.data_path, 'val.csv') 
        else:  
            mode = 'validation'
            anno_path = os.path.join(args.data_path, 'test.csv') 

        dataset = VideoClsDataset(
            anno_path=anno_path,
            data_path='/',
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
        nb_classes = 174

    elif args.data_set == 'UCF101':
        mode = None
        anno_path = None
        if is_train == True:
            mode = 'train'
            anno_path = os.path.join(args.data_path, 'train.csv')
        elif test_mode == True:
            mode = 'test'
            anno_path = os.path.join(args.data_path, 'val.csv') 
        else:  
            mode = 'validation'
            anno_path = os.path.join(args.data_path, 'test.csv') 

        dataset = VideoClsDataset(
            anno_path=anno_path,
            data_path='/',
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
        if is_train == True:
            mode = 'train'
            anno_path = os.path.join(args.data_path, 'train.csv')
        elif test_mode == True:
            mode = 'test'
            anno_path = os.path.join(args.data_path, 'val.csv') 
        else:  
            mode = 'validation'
            anno_path = os.path.join(args.data_path, 'test.csv') 

        dataset = VideoClsDataset(
            anno_path=anno_path,
            data_path='/',
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
        nb_classes = 51
    else:
        raise NotImplementedError()
    assert nb_classes == args.nb_classes
    print("Number of the class = %d" % args.nb_classes)

    return dataset, nb_classes


class BatchCollator(object):
    """
    From a list of samples from the dataset,
    returns the batched objectimages and targets.
    This should be passed to the DataLoader
    """
    def __init__(self, size_divisible=0):
        self.divisible = size_divisible
        self.size_divisible = self.divisible

    def __call__(self, batch):
        transposed_batch = list(zip(*batch))
        video_data = batch_different_videos(transposed_batch[0], self.size_divisible)
        boxes = transposed_batch[1]
        video_ids = transposed_batch[2]
        return video_data, boxes, video_ids


def build_ava_dataset(is_train, transforms):
    input_filename = 'ava_video_train_v2.2' if is_train else 'ava_video_val_v2.2'
    assert input_filename
    dataset_catalog = paths_catalog.DatasetCatalog
    data = dataset_catalog.get(input_filename)
    ava_args = data["args"]
    ava_args["remove_clips_without_annotations"] = is_train
    ava_args["frame_span"] = TransformsCfg.FRAME_NUM * TransformsCfg.FRAME_SAMPLE_RATE  # 64
    if not is_train:
        ava_args["box_thresh"] = 0.8  #
        ava_args["action_thresh"] = 0.  #
    else:
        ava_args["box_file"] = None

    ava_args["transforms"] = transforms
    dataset = AVAVideoDataset(**ava_args)
    return dataset

def build_kinetics_dataset(is_train,transforms):
    input_filename = 'ava_video_train_v2.2' if is_train else 'ava_video_val_v2.2'
    kinetics_args = {}
    if is_train:
        kinetics_args['kinetics_annfile'] = "/mnt/cache/xingsen/xingsen2/kinetics_train_v1.0.json"
        kinetics_args['box_file'] = None
        kinetics_args['transforms'] = transforms
        kinetics_args['remove_clips_without_annotations'] = True
        kinetics_args['frame_span'] = TransformsCfg.FRAME_NUM * TransformsCfg.FRAME_SAMPLE_RATE  # 64
    else:
        kinetics_args['kinetics_annfile'] = "/mnt/cache/xingsen/xingsen2/kinetics_val_v1.0.json"
        kinetics_args['box_file'] = '/mnt/cache/xingsen/xingsen2/kinetics_person_box.json'
        kinetics_args['box_thresh'] = 0.8
        kinetics_args['action_thresh'] = 0.
        kinetics_args['transforms'] = transforms
        kinetics_args['remove_clips_without_annotations'] = True
        kinetics_args['frame_span'] = TransformsCfg.FRAME_NUM * TransformsCfg.FRAME_SAMPLE_RATE  # 64
        kinetics_args['eval_file_paths'] = {
            "csv_gt_file" : '/mnt/cache/xingsen/xingsen2/kinetics_val_gt_v1.0.csv',
            "labelmap_file": "/mnt/cache/xingsen/ava_dataset/AVA/annotations/ava_action_list_v2.2_for_activitynet_2019.pbtxt",
            "exclusion_file": "/mnt/cache/xingsen/ava_dataset/AVA/annotations/ava_val_excluded_timestamps_v2.2.csv",
        }
    dataset = KineticsDataset(**kinetics_args)
    return dataset


def build_ak_dataset(is_train,transforms):
    input_filename = "ak_train" if is_train else "ak_val"
    assert input_filename
    dataset_catalog = paths_catalog.DatasetCatalog
    data_args = dataset_catalog.get(input_filename)
    if is_train:
        data_args['video_root'] = '/mnt/cache/xingsen/ava_dataset/AVA/clips/trainval/'
        data_args['ava_annfile'] = "/mnt/cache/xingsen/ava_dataset/AVA/annotations/ava_train_v2.2_min.json"
        data_args['kinetics_annfile'] = "/mnt/cache/xingsen/xingsen2/kinetics_train_v1.0.json"
        data_args['transforms'] = transforms
        data_args['remove_clips_without_annotations'] = True
        data_args['frame_span'] = TransformsCfg.FRAME_NUM * TransformsCfg.FRAME_SAMPLE_RATE  # 64
        data_args["ava_eval_file_path"] = {
                "csv_gt_file": "/mnt/cache/xingsen/ava_dataset/AVA/annotations/ava_train_v2.2.csv",
                "labelmap_file": "/mnt/cache/xingsen/ava_dataset/AVA/annotations/ava_action_list_v2.2_for_activitynet_2019.pbtxt",
                "exclusion_file": "/mnt/cache/xingsen/ava_dataset/AVA/annotations/ava_train_excluded_timestamps_v2.2.csv",
            }
    else:
        data_args['video_root'] = '/mnt/cache/xingsen/ava_dataset/AVA/clips/trainval/'
        data_args['ava_annfile'] = "/mnt/cache/xingsen/ava_dataset/AVA/annotations/ava_val_v2.2_min.json"
        data_args['kinetics_annfile'] = "/mnt/cache/xingsen/xingsen2/kinetics_val_v1.0.json"
        data_args['box_thresh'] = 0.8
        data_args['action_thresh'] = 0.
        data_args['transforms'] = transforms
        data_args['remove_clips_without_annotations'] = True
        data_args['kinetics_box_file'] = '/mnt/cache/xingsen/xingsen2/kinetics_person_box.json'
        data_args['ava_box_file'] = '/mnt/cache/xingsen/ava_dataset/AVA/boxes/ava_val_det_person_bbox.json'
        data_args['frame_span'] = TransformsCfg.FRAME_NUM * TransformsCfg.FRAME_SAMPLE_RATE  # 64
        data_args['eval_file_paths'] = {
                "csv_gt_file":'/mnt/cache/xingsen/xingsen2/ak_val_gt.csv',
                "labelmap_file": "/mnt/cache/xingsen/ava_dataset/AVA/annotations/ava_action_list_v2.2_for_activitynet_2019.pbtxt",
                "exclusion_file": "/mnt/cache/xingsen/ava_dataset/AVA/annotations/ava_val_excluded_timestamps_v2.2.csv",
            }
        data_args["ava_eval_file_path"] = {
                "csv_gt_file": "/mnt/cache/xingsen/ava_dataset/AVA/annotations/ava_val_v2.2.csv",
                "labelmap_file": "/mnt/cache/xingsen/ava_dataset/AVA/annotations/ava_action_list_v2.2_for_activitynet_2019.pbtxt",
                "exclusion_file": "/mnt/cache/xingsen/ava_dataset/AVA/annotations/ava_val_excluded_timestamps_v2.2.csv",
            }
        data_args['kinetics_eval_file_path'] = {
            "csv_gt_file" : '/mnt/cache/xingsen/xingsen2/kinetics_val_gt_v1.0.csv',
            "labelmap_file": "/mnt/cache/xingsen/ava_dataset/AVA/annotations/ava_action_list_v2.2_for_activitynet_2019.pbtxt",
            "exclusion_file": "/mnt/cache/xingsen/ava_dataset/AVA/annotations/ava_val_excluded_timestamps_v2.2.csv",
        }
        

    dataset = AKDataset(**data_args)
    return dataset
