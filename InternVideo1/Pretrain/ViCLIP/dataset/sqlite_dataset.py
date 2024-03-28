import logging
import os
import json
import sqlite3
import random
import io
from os.path import basename

import numpy as np

from dataset.base_dataset import ImageVideoBaseDataset
from dataset.text_prompt import kinetics_templates, imagenet_templates
from dataset.utils import load_anno, pre_text
from dataset.video_utils import VIDEO_READER_FUNCS
from utils.distributed import is_main_process

from dataset.serialize import get_local_rank, TorchShmSerializedList

logger = logging.getLogger(__name__)


def get_anno_by_id(cur: sqlite3.Cursor, id: int):
    """TODO: Docstring for get_anno_by_id.

    Args:
        cur (sqlite3.Cursor): The dataset cursor.
        id (int): The annotation id.

    Returns:

    """
    pass


class SQLiteImgTxtRetTrainDataset(ImageVideoBaseDataset):
    media_type = "image"

    def __init__(self, ann_file, transform, has_multi_vision_gt=False, num_epochs=1):
        super().__init__()

        if len(ann_file) == 3 and ann_file[2] == "video":
            self.media_type = "video"  
        elif ann_file[-1] == "only_video":
            self.media_type = "only_video"  
        else:
            self.media_type = "image"
        self.label_file, self.data_root = ann_file[:2]

        if '.json' in self.label_file:
            logger.info(f"Loading json file {self.label_file}")

            if get_local_rank() == 0:  # Only one rank need to read the file
                with open(self.label_file, 'r') as f:
                    annos = json.load(f)
                
                
                min_length = 2
                # filter out the caption with length less than 2
                captions = [pre_text(anno["caption"]) for anno in annos]
                captions_len = [len(caption.split()) for caption in captions]
                logger.info("Num samples: {}".format(len(captions)))
                logger.info("Num samples too short: {}".format(sum([l < min_length for l in captions_len])))
                annos = [anno for anno, l in zip(annos, captions_len) if l >= min_length]
                if num_epochs < 1:
                    logger.info(f"Num_epochs is set to {num_epochs}, randomly sampling the dataset")
                    num_to_sample = int(num_epochs * len(annos))
                    random.seed(42)
                    annos = random.sample(annos, num_to_sample)
            else:
                annos = []
            
            self.anno = TorchShmSerializedList(annos)

            self.num_examples = len(self.anno)
        else:
            if num_epochs < 1:
                raise ValueError("num_epochs must be >= 1 when using sql dataset")
            logger.info('Load sql file')
            self.con = sqlite3.connect("file:" + self.label_file + "?mode=ro", uri=True)
            self.cur = self.con.cursor()
            self.num_examples = self.cur.execute("SELECT COUNT(*) FROM annos").fetchone()[0]

        self.use_prompt = False
        if "imagenet" in self.label_file:
            self.use_prompt = True
            self.prompt = imagenet_templates
            logger.info(f"Use prompt for ImageNet")
            logger.info(self.prompt)

        self.transform = transform
        # each caption has multiple image as ground_truth, e.g., ssv2
        self.has_multi_vision_gt = has_multi_vision_gt
        assert not self.has_multi_vision_gt

    def get_anno(self, index):
        if '.json' in self.label_file:
            filename = self.anno[index][self.media_type]
            caption = self.anno[index]["caption"]
        else:
            query = f"SELECT * FROM annos WHERE id = {index};"
            res = self.cur.execute(query)
            id, filename, caption = res.fetchone()
        anno = {"image": os.path.join(self.data_root, filename), "caption": caption}
        if self.use_prompt:
            anno["caption"] = random.choice(self.prompt).format(anno["caption"])
        return anno

    def __len__(self):
        return self.num_examples

    def __getitem__(self, index):
        try:
            ann = self.get_anno(index)
            caption = pre_text(ann["caption"])
            # key = ann["caption"] if self.has_multi_vision_gt else basename(ann["image"])
            # if hasattr(self, "with_audio") and self.with_audio:
            #     image, audio, audio_mask, index = self.load_and_transform_media_data(index, ann["image"])
            #     return image, caption, audio, audio_mask, index
            # else:
            image, index = self.load_and_transform_media_data(index, ann["image"])
            return image, caption, index
        except Exception as e:
            logger.warning(f"Caught exception {e} when loading image {ann['image']}")
            index = np.random.randint(0, len(self))
            return self.__getitem__(index)


class SQLiteVidTxtRetTrainDataset(SQLiteImgTxtRetTrainDataset):
    media_type = "video"

    def __init__(
        self,
        ann_file,
        transform,
        num_frames=4,
        video_reader_type="decord",
        sample_type="rand",
        num_tries=3,
        is_paragraph_retrieval=False,
        has_multi_vision_gt=False,
        repeat_kinetics=1,
        num_epochs=1,
        # with_audio=False,
        # audio_length=20.495,
    ):
        super().__init__(ann_file, transform, has_multi_vision_gt, num_epochs)
        self.num_frames = num_frames
        self.video_reader_type = video_reader_type
        self.video_reader = VIDEO_READER_FUNCS[video_reader_type]
        self.sample_type = sample_type
        self.num_tries = num_tries
        self.is_paragraph_retrieval = is_paragraph_retrieval
        # self.with_audio = with_audio
        # self.audio_length = audio_length

        if is_paragraph_retrieval:
            raise ValueError(f"not implemented")

        self.use_prompt = False
        if "kinetics" in self.label_file and "raw" in self.label_file:
            logger.info(f"Before length: {len(self.anno)}")
            logger.info(f'Repeat kinetics for {repeat_kinetics} times')
            tmp = self.anno.copy()
            for _ in range(1, repeat_kinetics):
                self.anno.extend(tmp)
            self.num_examples = len(self.anno)
            logger.info(f"After length: {len(self.anno)}")
        elif "kinetics" in self.label_file:
            self.use_prompt = True
            self.prompt = kinetics_templates
            logger.info(f"Use prompt for Kinetics")
            logger.info(self.prompt)