import logging
import os
import random
try:
    from petrel_client.client import Client
except:
    Client = None
from torch.utils.data import Dataset
from .utils import load_image_from_path
from .av_utils import lazy_load_s3video

logger = logging.getLogger(__name__)


class BaseDataset(Dataset):
    """Base class that implements the image and video loading methods"""

    media_type = "video"

    def __init__(self):
        assert self.media_type in ["audio", "image", "video", "audio_video"]
        self.data_root = None
        self.data_root_prefix = ""
        self.anno_list = (
            None  # list(dict), each dict contains {"image": str, # image or video path}
        )
        self.transform = None
        self.audio_reader_type = None
        self.audio_sample_rate = None
        self.max_audio_length = None
        self.video_reader = None
        self.num_tries = None
        self.client = Client('~/petreloss.conf') if Client is not None else None
        self.trimmed30 = False

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def get_anno(self, index): # NOTE used for most ret_dataset
        """obtain the annotation for one media (video or image)

        Args:
            index (int): The media index.

        Returns: dict.
            - "image": the filename, video also use "image".
            - "caption": The caption for this file.

        """
        anno = self.anno_list[index]
        if self.data_root is not None:
            if self.media_type == "audio":
                anno["audio"] = self.data_root_prefix + os.path.join(self.data_root, anno["audio"])
            else:
                anno["image"] = self.data_root_prefix + os.path.join(self.data_root, anno["image"])
        return anno

    def load_and_transform_media_data(self, index, data_path):
        try:
            if self.media_type == "image":
                return self.load_and_transform_media_data_image(index, data_path)
            elif self.media_type == "audio":
                return self.load_and_transform_media_data_audio(index, data_path)
            elif self.media_type == "video":
                return self.load_and_transform_media_data_video(index, data_path)
            elif self.media_type == "audio_video":
                return self.load_and_transform_media_data_audio_video(index, data_path)
            else:
                raise NotImplementedError(self.media_type)
        except Exception as e:
            logger.info(f"Something wrong when read {data_path}")
            raise e

    def load_and_transform_media_data_image(self, index, data_path):
        if type(data_path) is dict:
            image = load_image_from_path(data_path["image"], client=self.client)
            if "crop_bbox" in data_path.keys():
                bbox = data_path["crop_bbox"]
                x0, y0, x1, y1 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
                image = image[:, :, y0:y1, x0:x1]
                
            image = self.transform(image)
        else:

            image = load_image_from_path(data_path, client=self.client)
            image = self.transform(image)
        return image, index
    
    def load_and_transform_media_data_video(self, index, data_path):
        if type(data_path) is dict:
            if data_path['read_clip_from_video']:
                if self.trimmed30:
                    raise NotImplementedError("lazy_load_s3video does not support trimmed30")
                frames = lazy_load_s3video(data_path['video'], self.num_frames, data_path['video_start_frame'], data_path['video_end_frame'], self.client)
            else:
                raise NotImplementedError(data_path)
        else:
            max_num_frames = self.max_num_frames if hasattr(self, "max_num_frames") else -1
            frames, frame_indices, video_duration = self.video_reader(
                data_path, self.num_frames, self.sample_type, 
                max_num_frames=max_num_frames, client=self.client,
                trimmed30=self.trimmed30
            )

        # NOTE shared aug for video frames
        frames = self.transform(frames)
        return frames, index

