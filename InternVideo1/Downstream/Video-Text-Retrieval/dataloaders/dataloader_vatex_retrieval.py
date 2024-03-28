from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import os
from torch.utils.data import Dataset
import numpy as np
import pickle
import json
from dataloaders.rawvideo_util import RawVideoExtractor
import io
from decord import VideoReader, cpu
import torch
from PIL import Image
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, InterpolationMode

try:
    from petrel_client.client import Client
    client = Client()

    # Disable boto logger
    import logging
    logging.getLogger('boto3').setLevel(logging.WARNING)
    logging.getLogger('botocore').setLevel(logging.WARNING)
    logging.getLogger('nose').setLevel(logging.WARNING)
except:
    client = None


class VATEX_DataLoader(Dataset):
    """VATEX dataloader"""
    def __init__(
            self,
            subset,
            data_path,
            features_path,
            tokenizer,
            max_words=30,
            feature_framerate=1.0,
            max_frames=100,
            image_resolution=224,
            frame_order=0,
            slice_framepos=0,
    ):
        self.data_path = data_path
        self.features_path = features_path
        self.feature_framerate = feature_framerate
        self.image_resolution = image_resolution
        self.max_words = max_words
        self.max_frames = max_frames
        self.tokenizer = tokenizer
        # 0: ordinary order; 1: reverse order; 2: random order.
        self.frame_order = frame_order
        assert self.frame_order in [0, 1, 2]
        # 0: cut from head frames; 1: cut from tail frames; 2: extract frames uniformly.
        self.slice_framepos = slice_framepos
        assert self.slice_framepos in [0, 1, 2]

        self.subset = subset
        assert self.subset in ["train", "val", "test"]
        video_id_path_dict = {}
        video_id_path_dict["train"] = os.path.join(self.data_path, "train_list.txt")
        video_id_path_dict["val"] = os.path.join(self.data_path, "test_list.txt")
        video_id_path_dict["test"] = os.path.join(self.data_path, "test_list.txt")
        with open(video_id_path_dict[self.subset], 'r') as fp:
            video_ids = [itm.strip() for itm in fp.readlines()]
# =============================================================================================

        video_json_path_dict = {}
        video_json_path_dict["train"] = os.path.join(self.data_path, "vatex_training_v1.0_ceph.json")
        video_json_path_dict["val"] = os.path.join(self.data_path, "vatex_validation_v1.0_ceph.json")
        video_json_path_dict["test"] = os.path.join(self.data_path, "vatex_validation_v1.0_ceph.json")
        captions = {}
        video_dict = {}
        with open(video_json_path_dict[self.subset], 'r') as f:
            json_data = json.load(f)
        for itm in json_data:
            video_id = itm["videoID"]
            if video_id not in video_ids:
                continue
            if "path" not in itm:
                continue
            path = itm["path"]
            caption = itm["enCap"]
            video_dict[video_id] = path
            captions[video_id] = caption

# ==============================================================================================================================

        self.video_dict = video_dict
        self.sample_len = 0
        self.sentences_dict = {}
        self.cut_off_points = []
        for video_id in video_ids:
            if video_id not in captions:
                continue
            for cap_txt in captions[video_id]:
                self.sentences_dict[len(self.sentences_dict)] = (video_id, cap_txt)
            self.cut_off_points.append(len(self.sentences_dict))

        print(f"sentence dict len: {len(self.sentences_dict)}")
        print(f"video dict len: {len(video_dict)}")
        print(f"video ids len: {len(video_ids)}")
        ## below variables are used to multi-sentences retrieval
        # self.cut_off_points: used to tag the label when calculate the metric
        # self.sentence_num: used to cut the sentence representation
        # self.video_num: used to cut the video representation
        self.multi_sentence_per_video = True    # !!! important tag for eval
        if self.subset == "val" or self.subset == "test":
            self.sentence_num = len(self.sentences_dict)
            self.video_num = len(video_dict)
            assert len(self.cut_off_points) == self.video_num

        self.sample_len = len(self.sentences_dict)
        self.rawVideoExtractor = RawVideoExtractor(framerate=feature_framerate, size=image_resolution)
        self.SPECIAL_TOKEN = {"CLS_TOKEN": "<|startoftext|>", "SEP_TOKEN": "<|endoftext|>",
                              "MASK_TOKEN": "[MASK]", "UNK_TOKEN": "[UNK]", "PAD_TOKEN": "[PAD]"}
        self.transform = Compose([
            Resize(image_resolution, interpolation=InterpolationMode.BICUBIC),
            CenterCrop(image_resolution),
            lambda image: image.convert("RGB"),
            ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
            # Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

    def __len__(self):
        return self.sample_len

    def _get_text(self, video_id, caption):
        k = 1
        choice_video_ids = [video_id]
        pairs_text = np.zeros((k, self.max_words), dtype=np.long)
        pairs_mask = np.zeros((k, self.max_words), dtype=np.long)
        pairs_segment = np.zeros((k, self.max_words), dtype=np.long)

        for i, video_id in enumerate(choice_video_ids):
            words = self.tokenizer.tokenize(caption)

            words = [self.SPECIAL_TOKEN["CLS_TOKEN"]] + words
            total_length_with_CLS = self.max_words - 1
            if len(words) > total_length_with_CLS:
                words = words[:total_length_with_CLS]
            words = words + [self.SPECIAL_TOKEN["SEP_TOKEN"]]

            input_ids = self.tokenizer.convert_tokens_to_ids(words)
            input_mask = [1] * len(input_ids)
            segment_ids = [0] * len(input_ids)
            while len(input_ids) < self.max_words:
                input_ids.append(0)
                input_mask.append(0)
                segment_ids.append(0)
            assert len(input_ids) == self.max_words
            assert len(input_mask) == self.max_words
            assert len(segment_ids) == self.max_words

            pairs_text[i] = np.array(input_ids)
            pairs_mask[i] = np.array(input_mask)
            pairs_segment[i] = np.array(segment_ids)

        return pairs_text, pairs_mask, pairs_segment, choice_video_ids

    def _get_rawvideo(self, choice_video_ids):
        video_mask = np.zeros((len(choice_video_ids), self.max_frames), dtype=np.long)
        max_video_length = [0] * len(choice_video_ids)

        # Pair x L x T x 3 x H x W
        video = np.zeros((len(choice_video_ids), self.max_frames, 1, 3,
                          self.rawVideoExtractor.size, self.rawVideoExtractor.size), dtype=np.float)

        for i, video_id in enumerate(choice_video_ids):
            video_path = self.video_dict[video_id]

            raw_video_data = self.rawVideoExtractor.get_video_data(video_path)
            raw_video_data = raw_video_data['video']

            if len(raw_video_data.shape) > 3:
                raw_video_data_clip = raw_video_data
                # L x T x 3 x H x W
                raw_video_slice = self.rawVideoExtractor.process_raw_data(raw_video_data_clip)
                if self.max_frames < raw_video_slice.shape[0]:
                    if self.slice_framepos == 0:
                        video_slice = raw_video_slice[:self.max_frames, ...]
                    elif self.slice_framepos == 1:
                        video_slice = raw_video_slice[-self.max_frames:, ...]
                    else:
                        sample_indx = np.linspace(0, raw_video_slice.shape[0] - 1, num=self.max_frames, dtype=int)
                        video_slice = raw_video_slice[sample_indx, ...]
                else:
                    video_slice = raw_video_slice

                video_slice = self.rawVideoExtractor.process_frame_order(video_slice, frame_order=self.frame_order)

                slice_len = video_slice.shape[0]
                max_video_length[i] = max_video_length[i] if max_video_length[i] > slice_len else slice_len
                if slice_len < 1:
                    pass
                else:
                    video[i][:slice_len, ...] = video_slice
            else:
                print("video path: {} error. video id: {}".format(video_path, video_id))

        for i, v_length in enumerate(max_video_length):
            video_mask[i][:v_length] = [1] * v_length

        return video, video_mask
        
    def _get_rawvideo_dec(self, choice_video_ids, s=None, e=None):
        # speed up video decode via decord.
        # video_mask = np.zeros(self.max_frames, dtype=np.long)
        video_mask = np.zeros((len(choice_video_ids), self.max_frames), dtype=np.long)
        
        # max_video_length = 0
        max_video_length = [0] * len(choice_video_ids)

        # T x 3 x H x W
        # video = np.zeros((self.max_frames, 3, self.image_resolution, self.image_resolution), dtype=np.float)
        video = np.zeros((len(choice_video_ids), self.max_frames, 1, 3,
                          self.image_resolution, self.image_resolution), dtype=np.float)

        if s is None:
            start_time, end_time = None, None
        else:
            start_time = int(s)
            end_time = int(e)
            start_time = start_time if start_time >= 0. else 0.
            end_time = end_time if end_time >= 0. else 0.
            if start_time > end_time:
                start_time, end_time = end_time, start_time
            elif start_time == end_time:
                end_time = start_time + 1
        # video_path = self.video_dict[video_id]
        for i, video_id in enumerate(choice_video_ids):
            video_path = self.video_dict[video_id]
            # video_path = os.path.join(self.features_path, "{}.mp4   ".format(video_id))
            if video_path.startswith("s3://"):
                video_bytes = client.get(video_path)
                if video_bytes is None:
                    print("Get video failed from {}".format(video_path))
                    continue
                video_path = io.BytesIO(video_bytes)
            vreader = VideoReader(video_path, ctx=cpu(0))
        
            fps = vreader.get_avg_fps()
            f_start = 0 if start_time is None else int(start_time * fps)
            f_end = int(min(1000000000 if end_time is None else end_time * fps, len(vreader) - 1))
            num_frames = f_end - f_start + 1
            if num_frames > 0:
                # T x 3 x H x W
                # sample_fps = int(self.video_framerate)
                sample_fps = int(self.feature_framerate)
                t_stride = int(round(float(fps) / sample_fps))

                all_pos = list(range(f_start, f_end + 1, t_stride))
                if len(all_pos) > self.max_frames:
                    sample_pos = [all_pos[_] for _ in np.linspace(0, len(all_pos) - 1, num=self.max_frames, dtype=int)]
                else:
                    sample_pos = all_pos

                patch_images = [Image.fromarray(f) for f in vreader.get_batch(sample_pos).asnumpy()]
                patch_images = torch.stack([self.transform(img) for img in patch_images])
                
                patch_images = patch_images.unsqueeze(1)
                
                slice_len = patch_images.shape[0]
                # max_video_length = max_video_length if max_video_length > slice_len else slice_len
                max_video_length[i] = max_video_length[i] if max_video_length[i] > slice_len else slice_len
                if slice_len < 1:
                    pass
                else:
                    video[i][:slice_len, ...] = patch_images
            else:
                print("video path: {} error. video id: {}".format(video_path, video_id))

        # video_mask[:max_video_length] = [1] * max_video_length
        for i, v_length in enumerate(max_video_length):
            video_mask[i][:v_length] = [1] * v_length

        #print(video.shape, video_mask.shape)
        return video, video_mask

    def __getitem__(self, idx):
        video_id, caption = self.sentences_dict[idx]

        pairs_text, pairs_mask, pairs_segment, choice_video_ids = self._get_text(video_id, caption)
        video, video_mask = self._get_rawvideo_dec(choice_video_ids)
        return pairs_text, pairs_mask, pairs_segment, video, video_mask