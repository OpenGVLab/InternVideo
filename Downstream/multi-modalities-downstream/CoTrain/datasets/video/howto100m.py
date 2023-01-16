from .video_base_dataset import BaseDataset
import torch as th
import pandas as pd
import os
import numpy as np
import random
import ffmpeg

import io
import decord
decord.bridge.set_bridge('torch')
from CoTrain.datasets import client


class HT100MDataset(BaseDataset):
    """HowTo100M Video-Text loader."""

    def __init__(self, *args, split="", **kwargs):
        assert split in ["train", "val", "test"]
        self.split = split

        if split == "train":
            names = ["howto100m_train"]
        elif split == "val":
            names = ["howto100m_val"]
        elif split == "test":
            names = ["howto100m_test"]
        super().__init__(*args, **kwargs, names=names, text_column_name="caption")

        self.metadata = None
        self._load_metadata()
        # for howto100
        self.min_time = 4.0
        self.size = 256
        self.fps = 2
        self.num_sec = self.num_frames / float(self.fps)
        self.crop_only = True
        if self.split == 'train':
            self.center_crop = False
        else:
            self.center_crop = True
        self.benchmark = False
        self.num_candidates = 1
        self.random_flip = True
        if "s3://" in self.data_dir:
            # Read from local
            self.caption_dir = os.path.join("./meta_data/howto100m", 'howto100m_csv')
            # self.caption_dir = os.path.join("./meta_data/howto100m", 'howto100m_csv')
        else:
            self.caption_dir = os.path.join(self.data_dir, 'howto100m_csv')
        print(names, ": ", len(self.metadata), "samples in total.")

    def _load_metadata(self):
        metadata_dir = './meta_data/howto100m_our'
        split_files = {
            'train': 'ht100_videos_split.csv',
            'val': 'ht100_videos_split_val.csv',            # there is no test
            'test': 'ht100_videos_split_val.csv'
        }
        target_split_fp = split_files[self.split]
        metadata = pd.read_csv(os.path.join(metadata_dir, target_split_fp), sep='\t')
        self.metadata = metadata["Name"]

    def read_frames_ffmpeg(self, video_path, start, end):
        start_seek = random.randint(int(start), int(max(start, end - self.num_sec)))
        cmd = (
            ffmpeg
            .input(video_path, ss=start_seek, t=self.num_sec + 0.01)
            .filter('fps', fps=self.fps)
        )
        if self.center_crop:
            aw, ah = 0.5, 0.5
        else:
            aw, ah = random.uniform(0, 1), random.uniform(0, 1)
        if self.crop_only:
            cmd = (
                cmd.crop('(iw - {})*{}'.format(self.size, aw),
                         '(ih - {})*{}'.format(self.size, ah),
                         str(self.size), str(self.size))
            )
        else:
            cmd = (
                cmd.crop('(iw - min(iw,ih))*{}'.format(aw),
                         '(ih - min(iw,ih))*{}'.format(ah),
                         'min(iw,ih)',
                         'min(iw,ih)')
                .filter('scale', self.size, self.size)
            )
        if self.random_flip and random.uniform(0, 1) > 0.5:
            cmd = cmd.hflip()
        out, _ = (
            cmd.output('pipe:', format='rawvideo', pix_fmt='rgb24').run(capture_stdout=True, quiet=True)
        )
        # print(np.frombuffer(out, np.uint8).shape)
        video = np.frombuffer(out, np.uint8).reshape([-1, self.size, self.size, 3])
        video_tensor = th.from_numpy(np.copy(video))
        video_tensor = video_tensor.permute(3, 0, 1, 2) + 0.01  # prevent all dark vide
        if video_tensor.shape[1] < self.num_frames:
            zeros = th.ones((3, self.num_frames - video_tensor.shape[1], self.size, self.size), dtype=th.uint8)
            video_tensor = th.cat((video_tensor, zeros), axis=1)
        return video_tensor[:, :self.num_frames]
    
    def read_frames_decord(self, video_path, start, end):
        # Build Decord conntainer
        video_bytes = client.get(video_path)
        assert video_bytes is not None, "Get video failed from {}".format(video_path)
        if isinstance(video_bytes, bytes):
            video_bytes = io.BytesIO(video_bytes)
        container = decord.VideoReader(video_bytes, ctx=decord.cpu(0))

        real_fps = container.get_avg_fps()
        start_frame, end_frame = real_fps * start, real_fps * end
        num_real_frames = self.num_frames / self.fps * real_fps
        start_seek = random.randint(int(start_frame), int(max(start_frame, end_frame - num_real_frames)))
        indexes = np.linspace(start_seek, start_seek + num_real_frames, self.num_frames, endpoint=False, dtype=int)
        indexes = [x for x in indexes if x < len(container)]
        assert len(indexes) > 0, "Failed to decode from {}".format(video_path)
        frames = container.get_batch(indexes)  # [T, H, W, C]
        _, H, W, _ = frames.shape
        assert self.crop_only
        if self.center_crop:
            aw, ah = 0.5, 0.5
        else:
            aw, ah = random.uniform(0, 1), random.uniform(0, 1)

        top = int((H - self.size) * ah)
        left = int((W - self.size) * aw)
        bottom, right = top + self.size, left + self.size
        frames = frames[:, top:bottom, left:right, :]
                
        if self.random_flip and random.uniform(0, 1) > 0.5:
            frames = frames.flip([-2])

        video_tensor = frames.permute(3, 0, 1, 2) + 0.01  # prevent all dark vide
        if video_tensor.shape[1] < self.num_frames:
            zeros = th.ones((3, self.num_frames - video_tensor.shape[1], self.size, self.size), dtype=th.uint8)
            video_tensor = th.cat((video_tensor, zeros), axis=1)
        # assert False, (video_tensor.min(), video_tensor.max())
        # assert False, video_tensor.shape
        return video_tensor[:, :self.num_frames]

    # time consuming > load video, where is the csv file? (cfs->ceph)
    def get_caption(self, caption):
        cap = pd.read_csv(caption)
        ind = random.randint(0, len(cap) - 1)
        text = cap['text'].values[ind]
        start, end = cap['start'].values[ind], cap['end'].values[ind]
        if end - start < self.min_time:
            diff = self.min_time - end + start
            start = max(0, start - diff / 2)
            end = start + self.min_time
        return text, start, end

    def get_text(self, sample):
        caption_csv = self.get_caption_path(sample)
        text, start, end = self.get_caption(caption_csv)
        # print(text)
        # TODO: May need to be improved for edge cases.
        encoding = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_text_len,
            return_special_tokens_mask=True,
        )
        return {"text": (text, encoding)}, int(start), int(end)

    def get_caption_path(self, sample):
        # example xx/xx/xx.mp4 -> xx.csv
        return os.path.join(self.caption_dir, sample.split('/')[-1].split('.')[0] + '.csv')

    def get_false_text(self, rep):
        random_index = random.randint(0, len(self.metadata) - 1)
        sample = self.metadata.iloc[random_index]
        caption_csv = self.get_caption_path(sample)
        text, start, end = self.get_caption(caption_csv)
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_text_len,
            return_special_tokens_mask=True,
        )
        return {f"false_text_{rep}": (text, encoding)}

    def _get_video_path(self, sample):
        rel_video_fp = sample
        if "s3://" in self.data_dir:
            rel_video_fp = sample.split("/")[-1]
            rel_video_fp = rel_video_fp.split(".")[0] + ".mp4" # All video on ceph is mp4
        full_video_fp = os.path.join(self.data_dir, rel_video_fp)
        return full_video_fp, rel_video_fp

    def get_raw_video(self, sample, begin, end):
        abs_fp, rel_fp = self._get_video_path(sample)
        if "s3://" in abs_fp:
            videos = self.read_frames_decord(abs_fp, begin, end).permute(1, 0, 2, 3)
        else:
            videos = self.read_frames_ffmpeg(abs_fp, begin, end).permute(1, 0, 2, 3)
        if videos is None:
            raise Exception("Invalid img!", rel_fp)
        else:
            return videos

    def get_video(self, sample, start, end):
        videos = self.get_raw_video(sample, start, end)
        videos_tensor = self.video_aug(videos, self.video_transform, byte=True)
        return videos_tensor

    def get_false_video(self, rep):
        random_index = random.randint(0, len(self.metadata) - 1)
        sample = self.metadata.iloc[random_index]
        caption_csv = self.get_caption_path(sample)
        _, start, end = self.get_caption(caption_csv)
        videos = self.get_raw_video(sample, start, end)
        videos_tensor = self.video_aug(videos, self.video_transform, byte=True)
        return {f"false_video_{rep}": videos_tensor}

    def get_suite(self, index):
        result = None
        max_try = 10
        try_time = 0
        while result is None:
            try_time += 1
            sample = self.metadata.iloc[index]
            try:
                ret = dict()
                text, start, end = self.get_text(sample)
                ret.update(text)
                videos_tensor = self.get_video(sample, start, end)
                ret.update({
                    "video": videos_tensor,
                    "vid_index": index,
                    "cap_index": index,
                    "raw_index": index,
                })
                ret.update({"replica": True if ret["cap_index"] > 0 else False})
                for i in range(self.draw_false_video):
                    ret.update(self.get_false_video(i))
                for i in range(self.draw_false_text):
                    ret.update(self.get_false_text(i))
                result = True
            except Exception as e:
                index = random.randint(0, len(self.metadata) - 1)
                exc = e
            if try_time > max_try:
                print(f"Exceed max time Error while read file idx {sample} in {self.names[0]} with error {exc}")
                try_time = 0
        return ret

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, index):
        return self.get_suite(index)
