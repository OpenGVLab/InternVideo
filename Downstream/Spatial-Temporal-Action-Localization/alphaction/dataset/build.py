import bisect
import copy

import torch.utils.data
from alphaction.utils.comm import get_world_size
from alphaction.utils.IA_helper import has_object
import alphaction.config.paths_catalog as paths_catalog

from . import datasets as D
from . import samplers

from .collate_batch import BatchCollator
from .transforms import build_transforms, build_object_transforms

def build_dataset(cfg, dataset_list, transforms, dataset_catalog, is_train=True, object_transforms=None):
    """
    Arguments:
        cfg: config object for the experiment.
        dataset_list (list[str]): Contains the names of the datasets, i.e.,
            ava_video_train_v2.2, ava_video_val_v2.2, etc..
        transforms (callable): transforms to apply to each (clip, target) sample.
        dataset_catalog (DatasetCatalog): contains the information on how to
            construct a dataset.
        is_train (bool): whether to setup the dataset for training or testing.
        object_transforms: transforms to apply to object boxes.
    """
    if not isinstance(dataset_list, (list, tuple)):
        raise RuntimeError(
            "dataset_list should be a list of strings, got {}".format(dataset_list)
        )
    datasets = []
    for dataset_name in dataset_list:
        data = dataset_catalog.get(dataset_name)
        factory = getattr(D, data["factory"])
        args = data["args"]
        if data["factory"] == "AVAVideoDataset":
            # for AVA, we want to remove clips without annotations
            # during training
            args["remove_clips_without_annotations"] = is_train
            args["frame_span"] = cfg.INPUT.FRAME_NUM*cfg.INPUT.FRAME_SAMPLE_RATE
            if not is_train:
                args["box_thresh"] = cfg.TEST.BOX_THRESH
                args["action_thresh"] = cfg.TEST.ACTION_THRESH
            else:
                # disable box_file when train, use only gt to train
                args["box_file"] = None
            if has_object(cfg.MODEL.IA_STRUCTURE):
                args["object_transforms"] = object_transforms
            else:
                args["object_file"] = None

        args["transforms"] = transforms
        # make dataset from factory
        dataset = factory(**args)
        datasets.append(dataset)

    # for testing, return a list of datasets
    if not is_train:
        return datasets

    # for training, concatenate all datasets into a single one
    dataset = datasets[0]
    if len(datasets) > 1:
        dataset = D.ConcatDataset(datasets)

    return [dataset]


def make_data_sampler(dataset, shuffle, distributed):
    if distributed:
        return samplers.DistributedSampler(dataset, shuffle=shuffle)
    if shuffle:
        sampler = torch.utils.data.sampler.RandomSampler(dataset)
    else:
        sampler = torch.utils.data.sampler.SequentialSampler(dataset)
    return sampler


def _quantize(x, bins):
    bins = copy.copy(bins)
    bins = sorted(bins)
    quantized = list(map(lambda y: bisect.bisect_right(bins, y), x))
    return quantized


def _compute_aspect_ratios(dataset):
    aspect_ratios = []
    for i in range(len(dataset)):
        video_info = dataset.get_video_info(i)
        aspect_ratio = float(video_info["height"]) / float(video_info["width"])
        aspect_ratios.append(aspect_ratio)
    return aspect_ratios


def make_batch_data_sampler(
        dataset, sampler, aspect_grouping, videos_per_batch, num_iters=None, start_iter=0, drop_last=False
):
    if aspect_grouping:
        if not isinstance(aspect_grouping, (list, tuple)):
            aspect_grouping = [aspect_grouping]
        aspect_ratios = _compute_aspect_ratios(dataset)
        group_ids = _quantize(aspect_ratios, aspect_grouping)
        batch_sampler = samplers.GroupedBatchSampler(
            sampler, group_ids, videos_per_batch, drop_uneven=drop_last
        )
    else:
        batch_sampler = torch.utils.data.sampler.BatchSampler(
            sampler, videos_per_batch, drop_last=drop_last
        )
    if num_iters is not None:
        batch_sampler = samplers.IterationBasedBatchSampler(
            batch_sampler, num_iters, start_iter
        )
    return batch_sampler


def make_data_loader(cfg, is_train=True, is_distributed=False, start_iter=0):
    num_gpus = get_world_size()
    if is_train:
        # for training
        videos_per_batch = cfg.SOLVER.VIDEOS_PER_BATCH
        assert (
                videos_per_batch % num_gpus == 0
        ), "SOLVER.VIDEOS_PER_BATCH ({}) must be divisible by the number "
        "of GPUs ({}) used.".format(videos_per_batch, num_gpus)
        videos_per_gpu = videos_per_batch // num_gpus
        shuffle = True
        drop_last = True
        num_iters = cfg.SOLVER.MAX_ITER
    else:
        # for testing
        videos_per_batch = cfg.TEST.VIDEOS_PER_BATCH
        assert (
                videos_per_batch % num_gpus == 0
        ), "TEST.VIDEOS_PER_BATCH ({}) must be divisible by the number "
        "of GPUs ({}) used.".format(videos_per_batch, num_gpus)
        videos_per_gpu = videos_per_batch // num_gpus
        shuffle = False if not is_distributed else True
        drop_last = False
        num_iters = None
        start_iter = 0

    # group images which have similar aspect ratio. In this case, we only
    # group in two cases: those with width / height > 1, and the other way around,
    # but the code supports more general grouping strategy
    aspect_grouping = [1] if cfg.DATALOADER.ASPECT_RATIO_GROUPING else []

    DatasetCatalog = paths_catalog.DatasetCatalog
    dataset_list = cfg.DATASETS.TRAIN if is_train else cfg.DATASETS.TEST

    # build dataset
    transforms = build_transforms(cfg, is_train)
    if has_object(cfg.MODEL.IA_STRUCTURE):
        object_transforms = build_object_transforms(cfg, is_train=is_train)
    else:
        object_transforms = None
    datasets = build_dataset(cfg, dataset_list, transforms, DatasetCatalog, is_train, object_transforms)

    # build sampler and dataloader
    data_loaders = []
    for dataset in datasets:
        sampler = make_data_sampler(dataset, shuffle, is_distributed)
        batch_sampler = make_batch_data_sampler(
            dataset, sampler, aspect_grouping, videos_per_gpu, num_iters, start_iter, drop_last
        )
        collator = BatchCollator(cfg.DATALOADER.SIZE_DIVISIBILITY)
        num_workers = cfg.DATALOADER.NUM_WORKERS
        data_loader = torch.utils.data.DataLoader(
            dataset,
            num_workers=num_workers,
            batch_sampler=batch_sampler,
            collate_fn=collator,
        )
        data_loaders.append(data_loader)
    if is_train:
        # during training, a single (possibly concatenated) data_loader is returned
        assert len(data_loaders) == 1
        return data_loaders[0]
    return data_loaders