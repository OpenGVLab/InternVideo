import logging
import os
import json
import random
import io
import torch
import numpy as np

from dataset.base_dataset import BaseDataset
from dataset.text_prompt import kinetics_templates, imagenet_templates
from dataset.utils import pre_text
from dataset.video_utils import VIDEO_READER_FUNCS
from dataset.serialize import get_local_rank, TorchShmSerializedList

logger = logging.getLogger(__name__)


class ImgTxtPtTrainDataset(BaseDataset):
    media_type = "image"

    def __init__(self, ann_file, transform, num_epochs=1):
        super().__init__()

        logger.info(f"ann_file: {ann_file}")

        self.media_type = ann_file.media_type
        self.label_file = ann_file.anno_path
        self.data_root = ann_file.data_root
        self.data_root_prefix = ann_file.get("data_root_prefix", "")
        self.min_caption_length = ann_file.get("min_caption_length", 2)
        self.caption_augmentation = ann_file.get("caption_augmentation", None)
        self.transform = transform
        # each caption has multiple image as ground_truth, e.g., ssv2
        self.has_multi_vision_gt = ann_file.get("has_multi_vision_gt", False)
        assert not self.has_multi_vision_gt

        self.crop_img = ann_file.get("crop_img", False)

        self.use_prompt = ann_file.get("prompt", "") != ""
        if self.use_prompt:
            if ann_file.prompt == "imagenet":
                self.prompt = imagenet_templates
                logger.info(f"Use prompt for ImageNet")
            elif ann_file.prompt == "kinetics":
                self.prompt = kinetics_templates
                logger.info(f"Use prompt for Kinetics")
            else:
                raise NotImplementedError(ann_file.prompt)
            logger.info(self.prompt)


        if self.use_prompt and self.caption_augmentation is not None:
            raise NotImplementedError("You can't use prompt because of multiple captions!")
        

        if '.json' in self.label_file:
            logger.info(f"Loading json file {self.label_file}")

            if get_local_rank() == 0:  # Only one rank need to read the file
                with io.BytesIO(self.client.get(self.label_file)) as f:
                # with open(self.label_file, 'r') as f:
                    annos = json.load(f)

                if ann_file.get("jump_filter", False):
                    logger.info("Jump filter!")
                else:
                    if self.caption_augmentation is not None:
                        # filter out the caption with length less than min_caption_length
                        new_annos = []
                        if self.media_type == "audio_video" and self.caption_augmentation.caption_sample_type == 'avs_all':
                            for anno in annos:
                                ok = True
                                if not anno['video'].endswith('.mp4'): 
                                    ok = False
                                for k in anno.keys():
                                    if "caption" in k and 'asr' not in k:
                                        tmp_c = pre_text(anno[k])
                                        if len(tmp_c.split()) < self.min_caption_length: 
                                            ok = False
                                            break
                                if ok:
                                    new_annos.append(anno)
                        elif self.caption_augmentation.caption_sample_type == 'uniform':
                            for anno in annos:
                                if "captions" in anno.keys():
                                    caption_key = "captions"
                                else:
                                    caption_key = "caption"
                                    
                                assert type(anno[caption_key]) is list, type(anno[caption_key])
                                caption_list = []
                                for c in anno[caption_key]:
                                    tmp_c = pre_text(c)
                                    if len(tmp_c.split()) >= self.min_caption_length:
                                        caption_list.append(tmp_c)

                                if len(caption_list) > 0:
                                    new_annos.append(anno)
                        else:
                            raise NotImplementedError(ann_file)
                        
                        logger.info(f"Num samples: {len(annos)}")
                        logger.info(f"Num samples not too short: {len(new_annos)} min_caption_length={self.min_caption_length}")
                        annos = new_annos
                    else:
                        # filter out the caption with length less than min_caption_length
                        captions = [pre_text(anno["caption"]) for anno in annos]
                        captions_len = [len(caption.split()) for caption in captions]
                        logger.info("Num samples: {}".format(len(captions)))
                        logger.info("Num samples too short: {}".format(sum([l < self.min_caption_length for l in captions_len])))
                        annos = [anno for anno, l in zip(annos, captions_len) if l >= self.min_caption_length]
                if num_epochs < 1:
                    raise NotImplementedError
            else:
                annos = []
            
            self.anno = TorchShmSerializedList(annos)
            self.num_examples = len(self.anno)
            logger.info(f"num_examples: {self.num_examples}")

        else:
            raise NotImplementedError("We need json file!!!")

    def __len__(self):
        return self.num_examples

    def get_caption(self, index):
        if '.json' in self.label_file:
            if self.caption_augmentation is not None:
                if self.caption_augmentation.caption_sample_type == 'avs_all':
                    caption_dict = {}
                    for k in self.anno[index].keys():
                        if 'caption' in k:
                            caption_dict[k] = self.anno[index][k]
                else:
                    if "captions" in self.anno[index].keys():
                        captions = self.anno[index]["captions"]
                    else:
                        captions = self.anno[index]["caption"]
            else:
                caption = self.anno[index]["caption"]
        else:
            raise NotImplementedError

        if self.caption_augmentation is not None:
            if self.caption_augmentation.caption_sample_type == 'uniform':
                caption = random.choice(captions)
            elif self.caption_augmentation.caption_sample_type == 'avs_all':
                caption = caption_dict
            else:
                raise NotImplementedError
        return caption
    
    def get_anno(self, index):
        assert self.media_type == 'image', self.media_type
        anno = {"caption": self.get_caption(index)}
        anno["image"] = self.data_root_prefix + os.path.join(self.data_root, self.anno[index]["image"])
        if self.use_prompt:
            anno["caption"] = random.choice(self.prompt).format(anno["caption"])
        if self.crop_img:
            anno["crop_bbox"] = self.anno[index]["crop_bbox"]
        return anno

    def pre_caption(self, caption):
        if type(caption) is str:
            return pre_text(caption)
        elif type(caption) is dict:
            assert self.caption_augmentation.caption_sample_type == 'avs_all'
            caption_dict = {}
            for k in caption.keys():
                caption_dict[k] = pre_text(caption[k])
            return caption_dict
        else:
            raise NotImplementedError(caption)

    def __getitem__(self, index):
        try:
            ann = self.get_anno(index)
            caption = self.pre_caption(ann["caption"])
            # key = ann["caption"] if self.has_multi_vision_gt else basename(ann["image"])
            if self.crop_img:
                data_path = {"image":ann["image"], "crop_bbox":ann["crop_bbox"]}
                image, index = self.load_and_transform_media_data(index, data_path)
            else:
                image, index = self.load_and_transform_media_data(index, ann["image"])
            return image, caption, index
        except Exception as e:
            logger.warning(f"Caught exception {e} when loading image {ann}")
            # raise e
            print(e)
            index = np.random.randint(0, len(self))
            return self.__getitem__(index)


class VidTxtPtTrainDataset(ImgTxtPtTrainDataset):
    media_type = "video"

    def __init__(
        self,
        ann_file,
        transform,
        num_frames=4,
        video_reader_type="decord",
        sample_type="rand",
        num_tries=3,
        num_epochs=1
    ):
        super().__init__(ann_file, transform, num_epochs)
        self.num_frames = num_frames
        self.video_reader_type = video_reader_type
        self.video_reader = VIDEO_READER_FUNCS[video_reader_type]
        self.sample_type = sample_type
        self.num_tries = num_tries

        self.is_paragraph_retrieval = ann_file.get("is_paragraph_retrieval", False)
        self.read_clip_from_video = ann_file.get("read_clip_from_video", False)
        
        if self.is_paragraph_retrieval:
            raise NotImplementedError

    def get_anno(self, index):
        assert self.media_type == "video", self.media_type
        anno = {"caption": self.get_caption(index)}
        anno["video"] = self.data_root_prefix + os.path.join(self.data_root, self.anno[index]["video"])
        if self.read_clip_from_video:
            anno["video_start_frame"] = self.anno[index]["video_start_frame"]
            anno["video_end_frame"] = self.anno[index]["video_end_frame"]
        if self.use_prompt:
            anno["caption"] = random.choice(self.prompt).format(anno["caption"])
        
        return anno

    def __getitem__(self, index):
        try:
            ann = self.get_anno(index)
            caption = self.pre_caption(ann["caption"])
            
            if self.read_clip_from_video:
                data_path = {
                    "video": ann["video"],
                    "video_start_frame": ann["video_start_frame"],
                    "video_end_frame": ann["video_end_frame"],
                    "read_clip_from_video": True
                }
            else:
                data_path = ann["video"]

            video, index = self.load_and_transform_media_data(index, data_path)

            return video, caption, index
        
        except Exception as e:
            logger.warning(f"Caught exception {e} when loading video {ann}")
            # raise e
            print(e)
            index = np.random.randint(0, len(self))
            return self.__getitem__(index)


class AudioVidTxtPtTrainDataset(VidTxtPtTrainDataset):
    media_type = "audio_video"

    def __init__(
        self,
        ann_file,
        transform,
        audio_sample_rate=16000, 
        audio_reader_type='torchaudio',
        max_audio_length=10,
        num_frames=4,
        video_reader_type="decord",
        sample_type="rand",
        num_tries=3,
        num_epochs=1
    ):
        super().__init__(ann_file, transform, num_epochs=num_epochs, num_frames=num_frames, video_reader_type=video_reader_type, sample_type=sample_type, num_tries=num_tries)

        assert self.media_type == 'audio_video', self.media_type
        self.audio_sample_rate = audio_sample_rate
        self.audio_reader_type = audio_reader_type
        self.max_audio_length = max_audio_length

        self.has_multi_audio_gt = ann_file.get("has_multi_audio_gt", False)
        self.read_audio_from_video = ann_file.get("read_audio_from_video", False)
        self.zero_audio_padding_for_video = ann_file.get("zero_audio_padding_for_video", False)

        self.now_tries = 0

    def get_anno(self, index):
        anno = {"caption": self.get_caption(index)}
        anno["video"] = self.data_root_prefix + os.path.join(self.data_root, self.anno[index]["video"])
        if self.read_clip_from_video:
            anno["video_start_frame"] = self.anno[index]["video_start_frame"]
            anno["video_end_frame"] = self.anno[index]["video_end_frame"]
        
        if "audio" in self.anno[index].keys():
            anno["audio"] = self.data_root_prefix + os.path.join(self.data_root, self.anno[index]["audio"])

        if self.use_prompt:
            anno["caption"] = random.choice(self.prompt).format(anno["caption"])
        
        return anno

    def __getitem__(self, index):
        try:
            ann = self.get_anno(index)
            caption = self.pre_caption(ann["caption"])
            data_path = {'video': ann["video"]}

            if self.read_clip_from_video:
                data_path["video_start_frame"] = ann["video_start_frame"]
                data_path["video_end_frame"] = ann["video_end_frame"]
            
            if "audio" in ann.keys():
                data_path["read_audio_from_video"] = False
                data_path["audio"] = ann["audio"]
            else:
                data_path["read_audio_from_video"] = self.read_audio_from_video

            data_path["read_clip_from_video"] = self.read_clip_from_video
            
            media, index = self.load_and_transform_media_data(index, data_path)
            self.now_tries = 0 

            audio = media[0]
            if audio is None and self.zero_audio_padding_for_video:
                logger.warning(f"No audio in {data_path}")
                media[0] = torch.zeros((998, 64), dtype=torch.float32)

            return media, caption, index
        
        except Exception as e:
            # print(e)
            if self.num_tries < self.now_tries:
                raise e
            else:
                self.now_tries += 1
            logger.warning(f"Caught exception {e} when loading audio-video {ann}")
            # logger.warning(f"Caught exception when loading audio-video {ann['video']}")
            
            index = np.random.randint(0, len(self))
            return self.__getitem__(index)


class AudioTxtPtTrainDataset(BaseDataset):
    media_type = "audio"

    def __init__(self, ann_file, transform, 
                audio_sample_rate=16000, 
                audio_reader_type='torchaudio',
                max_audio_length=10,
                num_tries=3,
                num_epochs=1):
        super().__init__()

        logger.info(f"ann_file: {ann_file}")

        self.media_type = ann_file.media_type
        self.label_file = ann_file.anno_path
        self.data_root = ann_file.data_root
        self.data_root_prefix = ann_file.get("data_root_prefix", "")
        self.min_caption_length = ann_file.get("min_caption_length", 2)
        self.caption_augmentation = ann_file.get("caption_augmentation", None)
        self.transform = transform
        self.audio_sample_rate = audio_sample_rate
        self.max_audio_length = max_audio_length
        self.audio_reader_type = audio_reader_type
        self.has_multi_audio_gt = ann_file.get("has_multi_audio_gt", False)
        assert not self.has_multi_audio_gt

        self.use_prompt = ann_file.get("prompt", "") != ""
        if self.use_prompt:
            if ann_file.prompt == "imagenet":
                self.prompt = imagenet_templates
                logger.info(f"Use prompt for ImageNet")
            elif ann_file.prompt == "kinetics":
                self.prompt = kinetics_templates
                logger.info(f"Use prompt for Kinetics")
            else:
                raise NotImplementedError(ann_file.prompt)
            logger.info(self.prompt)


        if self.use_prompt and self.caption_augmentation is not None:
            raise NotImplementedError("You can't use prompt because of multiple captions!")

        if '.json' in self.label_file:
            logger.info(f"Loading json file {self.label_file}")

            if get_local_rank() == 0:  # Only one rank need to read the file
                with io.BytesIO(self.client.get(self.label_file)) as f:
                # with open(self.label_file, 'r') as f:
                    annos = json.load(f)

                if ann_file.get("jump_filter", False):
                    logger.info("Jump filter!")
                else:
                    if self.caption_augmentation is not None:
                        # filter out the caption with length less than min_caption_length
                        new_annos = []
                        if self.caption_augmentation.caption_sample_type == 'uniform':
                            for anno in annos:
                                if "captions" in anno.keys():
                                    caption_key = "captions"
                                else:
                                    caption_key = "caption"
                                    
                                assert type(anno[caption_key]) is list, type(anno[caption_key])
                                caption_list = []
                                for c in anno[caption_key]:
                                    tmp_c = pre_text(c)
                                    if len(tmp_c.split()) >= self.min_caption_length:
                                        caption_list.append(tmp_c)

                                if len(caption_list) > 0:
                                    new_annos.append(anno)
                        else:
                            raise NotImplementedError(ann_file)
                        
                        logger.info(f"Num samples: {len(annos)}")
                        logger.info(f"Num samples not too short: {len(new_annos)} min_caption_length={self.min_caption_length}")
                        annos = new_annos
                    else:
                        # filter out the caption with length less than min_caption_length
                        captions = [pre_text(anno["caption"]) for anno in annos]
                        captions_len = [len(caption.split()) for caption in captions]
                        logger.info("Num samples: {}".format(len(captions)))
                        logger.info("Num samples too short: {}".format(sum([l < self.min_caption_length for l in captions_len])))
                        annos = [anno for anno, l in zip(annos, captions_len) if l >= self.min_caption_length]
                if num_epochs < 1:
                    raise NotImplementedError
            else:
                annos = []
            
            self.anno = TorchShmSerializedList(annos)
            self.num_examples = len(self.anno)
            logger.info(f"num_examples: {self.num_examples}")

        else:
            raise NotImplementedError("We need json file!!!")

    def __len__(self):
        return self.num_examples

    def get_caption(self, index):
        if '.json' in self.label_file:
            if self.caption_augmentation is not None:
                if "captions" in self.anno[index].keys():
                    captions = self.anno[index]["captions"]
                else:
                    captions = self.anno[index]["caption"]
            else:
                caption = self.anno[index]["caption"]
        else:
            raise NotImplementedError

        if self.caption_augmentation is not None:
            if self.caption_augmentation.caption_sample_type == 'uniform':
                caption = random.choice(captions)
            else:
                raise NotImplementedError
        return caption
    
    def get_anno(self, index):
        assert self.media_type == 'audio', self.media_type
        anno = {"caption": self.get_caption(index)}
        anno["audio"] = self.data_root_prefix + os.path.join(self.data_root, self.anno[index]["audio"])
        if self.use_prompt:
            anno["caption"] = random.choice(self.prompt).format(anno["caption"])

        return anno

    def pre_caption(self, caption):
        if type(caption) is str:
            return pre_text(caption)
        else:
            raise NotImplementedError(caption)
        
    def __getitem__(self, index):
        try:
            ann = self.get_anno(index)
            caption = self.pre_caption(ann["caption"])
            audio, index = self.load_and_transform_media_data(index, ann["audio"])
            return audio, caption, index
        except Exception as e:
            logger.warning(f"Caught exception {e} when loading audio {ann}")
            print(e)
            index = np.random.randint(0, len(self))
            return self.__getitem__(index)
