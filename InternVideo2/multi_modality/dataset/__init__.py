import logging
import torch
from torch.utils.data import ConcatDataset, DataLoader
from dataset.resample_concat_dataset import ResampleConcatDataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode
import copy

from dataset.sampler import StatefulDistributedSampler
from dataset.dataloader import MetaLoader, MetaLoader_rs # NOTE keep it
from dataset.ret_dataset import (ImgTxtRetTrainDataset,
                                 VidTxtRetTrainDataset,
                                 ImgTxtRetEvalDataset,
                                 VidTxtRetEvalDataset,
                                 VidTxtRetMCEvalDataset,
                                 VidTxtRetMCNewEvalDataset)
# from dataset.ret_dataset import (ImgTxtRetTrainDataset,
#                                  VidTxtRetTrainDataset,
#                                  ImgTxtRetEvalDataset,
#                                  VidTxtRetEvalDataset,
#                                  AudioVidTxtRetTrainDataset,
#                                  AudioVidTxtRetEvalDataset,
#                                  VidTxtRetMCEvalDataset,
#                                  AudioTxtRetTrainDataset,
#                                  AudioTxtRetEvalDataset)

from dataset.qa_dataset import ImageQADataset, VideoQADataset
from dataset.pt_dataset import (ImgTxtPtTrainDataset,
                                VidTxtPtTrainDataset,)
# from dataset.pt_dataset import (ImgTxtPtTrainDataset,
#                                 VidTxtPtTrainDataset,
#                                 AudioVidTxtPtTrainDataset,
#                                 AudioTxtPtTrainDataset)

logger = logging.getLogger(__name__)

def get_media_type(dataset_config):
    return dataset_config['media_type']

def get_dataset_cls(dataset_type, media_type, data_cfg):
    if dataset_type == "pt_train":
        if media_type == "image":
            dataset_cls = ImgTxtPtTrainDataset
        elif media_type == "video":
            dataset_cls = VidTxtPtTrainDataset
        # elif media_type == "audio_video":
        #     dataset_cls = AudioVidTxtPtTrainDataset
        # elif media_type == "audio":
        #     dataset_cls = AudioTxtPtTrainDataset
        else:
            raise NotImplementedError(f"dataset_type={dataset_type}, media_type={media_type}")
    elif dataset_type == "ret_train":
        if media_type == "image":
            dataset_cls = ImgTxtRetTrainDataset
        elif media_type == "video":
            dataset_cls = VidTxtRetTrainDataset
        # elif media_type == 'audio':
        #     dataset_cls = AudioTxtRetTrainDataset
        # elif media_type == "audio_video":
        #     dataset_cls = AudioVidTxtRetTrainDataset
        else:
            raise NotImplementedError(f"dataset_type={dataset_type}, media_type={media_type}")
    elif dataset_type == "ret_eval":
        if media_type == "image":
            dataset_cls = ImgTxtRetEvalDataset
        elif media_type == "video":
            dataset_cls = VidTxtRetEvalDataset
        # elif media_type == "audio":
        #     dataset_cls = AudioTxtRetEvalDataset
        # elif media_type == "audio_video":
        #     dataset_cls = AudioVidTxtRetEvalDataset
        else:
            raise NotImplementedError(f"dataset_type={dataset_type}, media_type={media_type}")
    elif dataset_type in ["qa_train", 'qa_eval']:
        if media_type == "image":
            dataset_cls = ImageQADataset
        elif media_type == "video":
            dataset_cls = VideoQADataset
        else:
            raise NotImplementedError(f"dataset_type={dataset_type}, media_type={media_type}")
    else:
        raise NotImplementedError(f"dataset_type={dataset_type}, media_type={media_type}")
    
    print(f"\033[31m dataset_type: {dataset_type} media_type: {media_type} dataset_cls: {dataset_cls}\033[0m")
    logger.info(f"dataset_type: {dataset_type} media_type: {media_type} dataset_cls: {dataset_cls}")

    return dataset_cls

def get_train_transform(config, train_file):
    vision_enc_name = config.model.vision_encoder.name
    if "internvideo" in vision_enc_name or "vit" in vision_enc_name or "umt" in vision_enc_name:
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
    elif "clip" in vision_enc_name:
        mean = (0.48145466, 0.4578275, 0.40821073)
        std = (0.26862954, 0.26130258, 0.27577711)
    else:
        raise NotImplementedError(vision_enc_name)

    normalize = transforms.Normalize(mean, std)

    # loaded images and videos are torch.Tensor of torch.uint8 format,
    # ordered as (T, 1 or 3, H, W) where T=1 for image
    type_transform = transforms.Lambda(lambda x: x.float().div(255.0))

    if config.inputs.video_input.random_aug:
        aug_transform = transforms.RandAugment()
    else:
        aug_transform = transforms.Lambda(lambda x: x)

    train_transform = transforms.Compose(
        [
            aug_transform,
            transforms.RandomResizedCrop(
                config.inputs.image_res,
                scale=(0.5, 1.0),
                interpolation=InterpolationMode.BICUBIC,
            ),
            transforms.RandomHorizontalFlip(),
            type_transform,
            normalize,
        ]
    )

    return train_transform

def get_test_transform(config, test_file):
    vision_enc_name = config.model.vision_encoder.name
    if "internvideo" in vision_enc_name or "vit" in vision_enc_name or "umt" in vision_enc_name:
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
    elif "clip" in vision_enc_name:
        mean = (0.48145466, 0.4578275, 0.40821073)
        std = (0.26862954, 0.26130258, 0.27577711)
    else:
        raise NotImplementedError(vision_enc_name)

    normalize = transforms.Normalize(mean, std)

    # loaded images and videos are torch.Tensor of torch.uint8 format,
    # ordered as (T, 1 or 3, H, W) where T=1 for image
    type_transform = transforms.Lambda(lambda x: x.float().div(255.0))

    test_transform = transforms.Compose(
        [
            transforms.Resize(
                (config.inputs.image_res, config.inputs.image_res),
                interpolation=InterpolationMode.BICUBIC,
            ),
            type_transform,
            normalize,
        ]
    )
    return test_transform


def create_dataset(dataset_type, config):
    ##########################################################
    # Shared setting for all datasets
    if config.inputs.get('video_input', None) is not None:
        video_reader_type = config.inputs.video_input.get("video_reader_type", "decord")
        video_only_dataset_kwargs_train = dict(
            video_reader_type=video_reader_type,
            sample_type=config.inputs.video_input.sample_type,
            num_frames=config.inputs.video_input.num_frames,
            num_tries=10,  # false tolerance
        )
        video_only_dataset_kwargs_eval = dict(
            video_reader_type=video_reader_type,
            sample_type=config.inputs.video_input.sample_type_test,
            num_frames=config.inputs.video_input.num_frames_test,
            num_tries=1,  # we want to have predictions for all videos
        )
    else:
        logger.warn("Make sure that you don't need video input!!!")
    if config.inputs.get('audio_input', None) is not None:
        audio_reader_type = config.inputs.audio_input.get("audio_reader_type", "torchaudio")
        audio_only_dataset_kwargs_train = dict(
            audio_reader_type=audio_reader_type,
            audio_sample_rate=config.inputs.audio_input.get('audio_sample_rate', 16000),
            max_audio_length=config.inputs.audio_input.get('max_audio_length', 10),
            num_tries=10,
        )
        audio_only_dataset_kwargs_eval = dict(
            audio_reader_type=audio_reader_type,
            audio_sample_rate=config.inputs.audio_input.get('audio_sample_rate_test', 16000),
            max_audio_length=config.inputs.audio_input.get('max_audio_length', 10),
            num_tries=1,
        )
    else:
        logger.warn("Make sure that you don't need audio input!!!")


    if dataset_type == "pt_train":
        # convert to list of lists
        train_files = (
            [config.train_file] if isinstance(config.train_file, dict) else config.train_file
        )
        train_media_types = sorted(list({get_media_type(e) for e in train_files}))

        train_datasets = []
        for m in train_media_types:
            
            # dataset of the same media_type will be mixed in a single Dataset object
            _train_files = [e for e in train_files if get_media_type(e) == m]

            datasets = []
            sample_weights = []
            for train_file in _train_files:
                dataset_cls = get_dataset_cls(dataset_type=dataset_type, media_type=m, data_cfg=train_file)
                if m == "audio":
                    train_transform = None
                else:
                    train_transform = get_train_transform(config, train_file)
                dataset_kwargs = dict(
                    ann_file=train_file,
                    transform=train_transform,
                    num_epochs=config.scheduler.epochs)

                if m == "audio_video":
                    dataset_kwargs.update(video_only_dataset_kwargs_train)
                    dataset_kwargs.update(audio_only_dataset_kwargs_train)
                elif m == "video":
                    dataset_kwargs.update(video_only_dataset_kwargs_train)
                elif m == 'audio':
                    dataset_kwargs.update(audio_only_dataset_kwargs_train)
                elif m != 'image':
                    raise NotImplementedError(m)
                logger.info(f"dataset_type={dataset_type}, train_file={train_file}")
                logger.info(dataset_kwargs)
                logger.info('train_transform:')
                logger.info(str(train_transform))

                datasets.append(dataset_cls(**dataset_kwargs))
                sample_weights.append(train_file.get("sample_weight", 1))
                # assert train_file.get("sample_weight", 1) == 1, train_file

            if sum(sample_weights) > len(sample_weights):
                logger.info(f'Use ResampleConcatDataset for {m}, sample_weights={sample_weights}')
                dataset = ResampleConcatDataset(datasets, sample_weights=sample_weights)
            else:
                logger.info(f'Use ConcatDataset for {m}')
                dataset = ConcatDataset(datasets)

            train_datasets.append(dataset)

        return train_datasets

    elif dataset_type == "ret_train":
        assert isinstance(config.train_file, dict), config.train_file
        train_transform = get_train_transform(config, config.train_file)
        dataset_cls = get_dataset_cls(dataset_type=dataset_type,
                                      media_type=config.train_file.media_type,
                                      data_cfg=config.train_file)
        if config.train_file.media_type == "video":
            dataset_kwargs = dict(
                ann_file=config.train_file,
                transform=train_transform)
            dataset_kwargs.update(video_only_dataset_kwargs_train)
        elif config.train_file.media_type == 'audio':
            dataset_kwargs = dict(
                ann_file=config.train_file,
                transform=None)
            dataset_kwargs.update(audio_only_dataset_kwargs_train)
        elif config.train_file.media_type == 'audio_video':
            dataset_kwargs = dict(
                ann_file=config.train_file,
                transform=train_transform)
            dataset_kwargs.update(video_only_dataset_kwargs_train)
            dataset_kwargs.update(audio_only_dataset_kwargs_train)
        else:
            raise NotImplementedError(config.train_file.media_type)
        
        logger.info(f"dataset_type={dataset_type}, train_file={config.train_file}")
        logger.info(dataset_kwargs)
        logger.info('train_transform:')
        logger.info(str(train_transform))

        return [dataset_cls(**dataset_kwargs)]
    
    elif dataset_type == "qa_train":
        assert type(config.train_file) is dict, f"assuming single train media type but get {config.train_file}"
        
        media_type = get_media_type(config.train_file[0])  # assuming single train media type
        if media_type == "audio":
            train_transform = None
        else:
            train_transform = get_train_transform(config, config.train_file)
            
        dataset_cls = get_dataset_cls(dataset_type=dataset_type,
                                      media_type=media_type,
                                      data_cfg=config.train_file)
        dataset_kwargs = dict(
            ann_file=config.train_file, transform=train_transform, eos=config.eos, mode="train"
        )
        if media_type == "video":
            dataset_kwargs.update(video_only_dataset_kwargs_train)
        train_dataset = dataset_cls(**dataset_kwargs)

        logger.info(f"dataset_type={dataset_type}, train_file={config.train_file}")
        logger.info(dataset_kwargs)
        logger.info('train_transform:')
        logger.info(str(train_transform))

        return train_dataset
    
    elif dataset_type in ["pt_eval", "ret_eval", "qa_eval"]:
        test_datasets = []
        test_dataset_names = []
        # multiple test datasets, all separate
        for name, data_cfg in config.test_file.items():
            media_type = get_media_type(data_cfg)
            test_dataset_names.append(name)
            test_transform = get_test_transform(config, data_cfg)

            if dataset_type == "qa_eval" or (dataset_type == "pt_eval" and "_qa_" in name):
                test_dataset_cls = get_dataset_cls(dataset_type='qa_eval',
                                                   media_type=media_type,
                                                   data_cfg=data_cfg)
                dataset_kwargs = dict(
                    ann_file=data_cfg,
                    transform=test_transform,
                    eos=config.eos,
                    mode="eval",
                    answer_list=config.answer_list,
                )
                if media_type == "video":
                    dataset_kwargs.update(video_only_dataset_kwargs_eval)
                else:
                    raise NotImplementedError(f"media_type={media_type}")
            else: # ret
                test_dataset_cls = get_dataset_cls(dataset_type='ret_eval',
                                                   media_type=media_type,
                                                   data_cfg=data_cfg)
                if media_type == "video":
                    dataset_kwargs = dict(
                        ann_file=data_cfg,
                        transform=test_transform
                    )
                    if "hmdb" in name or 'frame' in name: # read image for video
                        dataset_kwargs["video_reader_type"] = 'img'
                    dataset_kwargs.update(video_only_dataset_kwargs_eval)
                elif media_type == "audio":
                    dataset_kwargs = dict(
                        ann_file=data_cfg,
                        transform=None)
                    dataset_kwargs.update(audio_only_dataset_kwargs_eval)
                elif media_type == 'audio_video':
                    dataset_kwargs = dict(
                        ann_file=data_cfg,
                        transform=test_transform)
                    dataset_kwargs.update(video_only_dataset_kwargs_eval)
                    dataset_kwargs.update(audio_only_dataset_kwargs_eval)
                elif media_type != 'image':
                    raise NotImplementedError(f"media_type={media_type}")
                
            logger.info(f"dataset_type={dataset_type}, test_file={data_cfg}")
            logger.info(dataset_kwargs)
            logger.info('test_transform:')
            logger.info(str(test_transform))

            test_datasets.append(test_dataset_cls(**dataset_kwargs))
        return test_datasets, test_dataset_names
    
    elif dataset_type == "mc_test":
        test_transform = get_test_transform(config, config.test_file.mc_test)
        dataset_kwargs = dict(ann_file=[config.test_file.mc_test], transform=test_transform)
        dataset_kwargs.update(video_only_dataset_kwargs_eval)

        logger.info(f"dataset_type={dataset_type}, test_file={config.test_file}")
        logger.info(dataset_kwargs)
        logger.info('test_transform:')
        logger.info(str(test_transform))
        
        return VidTxtRetMCEvalDataset(**dataset_kwargs)
    
    elif dataset_type == "mc_new_test":
        test_transform = get_test_transform(config, config.test_file.mc_test)
        dataset_kwargs = dict(ann_file=[config.test_file.mc_test], transform=test_transform)
        dataset_kwargs.update(video_only_dataset_kwargs_eval)

        logger.info(f"dataset_type={dataset_type}, test_file={config.test_file}")
        logger.info(dataset_kwargs)
        logger.info('test_transform:')
        logger.info(str(test_transform))
        
        return VidTxtRetMCNewEvalDataset(**dataset_kwargs)
    
    else:
        raise NotImplementedError(f"dataset_type={dataset_type}")

def vqa_collate_fn(batch):
    image_list, question_list, answer_list, weight_list, n = [], [], [], [], []
    for image, question, answer, weights in batch:
        image_list.append(image)
        question_list.append(question)
        weight_list += weights
        answer_list += answer
        n.append(len(answer))
    return (
        torch.stack(image_list, dim=0),
        question_list,
        answer_list,
        torch.Tensor(weight_list),
        n,
    )


def create_sampler(datasets, shuffles, num_tasks, global_rank):
    samplers = []
    for dataset, shuffle in zip(datasets, shuffles):
        sampler = torch.utils.data.DistributedSampler(
            dataset, num_replicas=num_tasks, rank=global_rank, shuffle=shuffle
        )
        samplers.append(sampler)
    return samplers

def create_stateful_sampler(datasets, batch_size):
    samplers = []
    for dataset, bs in zip(datasets, batch_size):
        sampler = StatefulDistributedSampler(dataset, batch_size=bs)
        samplers.append(sampler)
    return samplers

def create_loader(datasets, samplers, batch_size, num_workers, is_trains, collate_fns):
    loaders = []
    for dataset, sampler, bs, n_worker, is_train, collate_fn in zip(
        datasets, samplers, batch_size, num_workers, is_trains, collate_fns
    ):
        if is_train:
            shuffle = sampler is None
            drop_last = True
            pin_memory = True
            persistent_workers = True if n_worker > 0 else False
        else:
            shuffle = False
            drop_last = False
            pin_memory = False
            persistent_workers = False
        loader = DataLoader(
            dataset,
            batch_size=bs,
            num_workers=n_worker,
            pin_memory=pin_memory,
            sampler=sampler,
            shuffle=shuffle,
            collate_fn=collate_fn,
            drop_last=drop_last,
            persistent_workers=persistent_workers,
        )
        loaders.append(loader)
    return loaders


def iterate_dataloaders(dataloaders):
    """Alternatively generate data from multiple dataloaders,
    since we use `zip` to concat multiple dataloaders,
    the loop will end when the smaller dataloader runs out.

    Args:
        dataloaders List(DataLoader): can be a single or multiple dataloaders
    """
    for data_tuples in zip(*dataloaders):
        for idx, data in enumerate(data_tuples):
            yield dataloaders[idx].dataset.media_type, data
