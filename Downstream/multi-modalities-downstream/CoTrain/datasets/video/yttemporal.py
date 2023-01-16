from .video_base_dataset import BaseDataset, sample_frames, video_clip_reader, clean_subtitles, align_using_dtw
import torch as th
import pandas as pd
import os
import numpy as np
import random
import ffmpeg
import json
import ftfy


class YTTemporalDataset(BaseDataset):
    """YTTemporal Video-Text loader."""

    def __init__(self, *args, split="", **kwargs):
        assert split in ["train", "val", "test"]
        self.split = split

        if split == "train":
            names = ["yttemporal_train"]
        elif split == "val":
            names = ["yttemporal_val"]
        elif split == "test":
            names = ["yttemporal_test"]
        super().__init__(*args, **kwargs, names=names, text_column_name="caption")

        self.metadata = None
        self._load_metadata()
        self.min_time = 4.0
        self.size = 224
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
        self._load_metadata()

    def _load_metadata(self):
        metadata_dir = './meta_data/yttemporal'
        split_files = {
            'train': 'train_success_2000000.csv',  # _1000000
            'val': 'val_success.csv',            # there is no test
            'test': 'val_success.csv'
        }
        target_split_fp = split_files[self.split]
        metadata = pd.read_csv(os.path.join(metadata_dir, target_split_fp), sep='\t')
        self.metadata = metadata["Name"]

    def read_frames_ffmpeg(self, video_path, start, end):
        start_seek = start
        cmd = (
            ffmpeg
            .input(video_path, ss=start_seek, t=end-start + 0.01)
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
            cmd.output('pipe:', format='rawvideo', pix_fmt='rgb24')
            .run(capture_stdout=True, quiet=True)
        )
        video = np.frombuffer(out, np.uint8).reshape([-1, self.size, self.size, 3])
        video_tensor = th.from_numpy(np.copy(video))
        video_tensor = video_tensor.permute(3, 0, 1, 2) + 0.01
        # print(video_tensor.size())
        # print(video_tensor)
        if video_tensor.shape[1] < self.num_frames:
            zeros = th.ones((3, self.num_frames - video_tensor.shape[1], self.size, self.size), dtype=th.uint8)
            video_tensor = th.cat((video_tensor, zeros), axis=1)
        # # uniform n frames
        # frame_indexs = sample_frames(self.num_frames, video_tensor.size(1))
        # out_tensors = th.ones((3, self.num_frames, self.size, self.size), dtype=th.uint8)
        # for i in range(self.num_frames):
        #     out_tensors[:, i] = video_tensor[:, frame_indexs[i]]
        # print(out_tensors)
        # return out_tensors
        return video_tensor[:, :self.num_frames]

    # # sample fix number of words
    # def get_caption(self, caption_csv, words_len=32):
    #     with open(caption_csv, 'r') as f:
    #         cap = json.load(f)
    #     # random choice words_len words
    #     video_len = int(cap["info"]["duration"])
    #     all_text = cap["subtitles"]  # [{'word': 'hey', 'time': 0.0}, {'word': 'guys', 'time': 0.149}]
    #     word_count = len(all_text)
    #
    #     # clean noisy asr text
    #     all_text = clean_subtitles(all_text)
    #     vtt = pd.DataFrame(all_text)
    #     denoised_word_by_word = []
    #     for x in cap['denoised']:
    #         # Ftfy just in case
    #         cleanasr = ftfy.ftfy(x['cleanasr'])
    #         denoised_word_by_word += cleanasr.split(' ')
    #     # Align
    #     vtt['denoised'] = align_using_dtw(vtt['word'], denoised_word_by_word)
    #     max_word = min(word_count - 1, words_len)
    #     begin_word_index = random.randint(0, word_count - max_word)
    #     text = ""
    #     for i in range(max_word):
    #         text += vtt['denoised'][begin_word_index + i] + ' '
    #     start = float(all_text[begin_word_index]['time'])
    #     end = float(all_text[min(word_count-1, begin_word_index + max_word)]['time'])
    #     # print(text, start, end)
    #     return text, start, end, video_len

    # sample fix video length
    def get_caption(self, caption_csv):
        with open(caption_csv, 'r') as f:
            cap = json.load(f)
        # random choice 10s-15s video clips
        video_len = int(cap["info"]["duration"])
        start = random.randint(0, max(1, video_len-15)) + random.random()
        clip_len = random.randint(10, 15)
        end = min(video_len-1, start + clip_len)
        all_text = cap["subtitles"]  # [{'word': 'hey', 'time': 0.0}, {'word': 'guys', 'time': 0.149}]
        # clean noisy asr text
        all_text = clean_subtitles(all_text)
        vtt = pd.DataFrame(all_text)
        denoised_word_by_word = []
        for x in cap['denoised']:
            # Ftfy just in case
            cleanasr = ftfy.ftfy(x['cleanasr'])
            denoised_word_by_word += cleanasr.split(' ')
        # Align
        vtt['denoised'] = align_using_dtw(vtt['word'], denoised_word_by_word)
        text = ""
        origin_text = ""
        for index, item in enumerate(all_text):
            if float(item['time']) > start and float(item['time']) < end:
                text += vtt['denoised'][index] + " "
                origin_text += item['word'] + " "
        # print(text)
        # print(origin_text)
        if len(text) < 10:
            Exception(IndexError)
        if end - start < self.min_time:
            diff = self.min_time - end + start
            start = max(0, start - diff / 2)
            end = start + self.min_time
        return text, start, end, video_len

    def get_text(self, sample):
        caption_csv = self.get_caption_path(sample)
        text, start, end, duration = self.get_caption(caption_csv)
        # print(text, start, end)
        # print(text)
        # TODO: May need to be improved for edge cases.
        encoding = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_text_len,
            return_special_tokens_mask=True,
        )
        return {"text": (text, encoding)}, start, end, duration

    def get_caption_path(self, sample):
        # YTTemporal/videos/subset_87/data/xx.mp4 -> YTTemporal/videos/subset_87/annotations/xx.csv
        return os.path.join(self.data_dir, 'videos', sample.split('/')[0], 'annotations', sample.split('/')[-1][:-4] + '.json')

    def get_false_text(self, rep):
        random_index = random.randint(0, len(self.metadata) - 1)
        sample = self.metadata.iloc[random_index]
        caption_csv = self.get_caption_path(sample)
        text, start, end = self.get_caption(caption_csv)
        encoding = self.tokenizer(
            text,
            # padding="max_length",
            truncation=True,
            max_length=self.max_text_len,
            return_special_tokens_mask=True,
        )
        return {f"false_text_{rep}": (text, encoding)}

    def _get_video_path(self, sample):
        rel_video_fp = sample
        full_video_fp = os.path.join(self.data_dir, 'videos', rel_video_fp)
        return full_video_fp, rel_video_fp

    def get_raw_video(self, sample, begin, end, duration):
        abs_fp, rel_fp = self._get_video_path(sample)
        # print(abs_fp, rel_fp)
        # imgs = self.read_frames_ffmpeg(abs_fp, begin, end).permute(1, 0, 2, 3)
        videos = video_clip_reader(abs_fp, begin, end, duration, self.num_frames)
        if videos.size(0) != self.num_frames:
            raise Exception("video length not enough!", rel_fp)
        if videos is None:
            raise Exception("Invalid img!", rel_fp)
        else:
            return videos

    def get_video(self, sample, start, end, duration):
        videos = self.get_raw_video(sample, start, end, duration)
        videos_tensor = self.video_aug(videos, self.video_transform)
        return videos_tensor

    def get_false_video(self, rep):
        random_index = random.randint(0, len(self.metadata) - 1)
        sample = self.metadata.iloc[random_index]
        caption_csv = self.get_caption_path(sample)
        _, start, end, duration = self.get_caption(caption_csv)
        videos = self.get_raw_video(sample, start, end, duration)
        videos_tensor = self.video_aug(videos, self.video_transform)
        return {f"false_video_{rep}": videos_tensor}

    def get_suite(self, index):
        result = None
        max_try = 5
        try_time = 0
        while result is None:
            try_time += 1
            sample = self.metadata.iloc[index]
            # try:
            ret = dict()
            text, start, end, duration = self.get_text(sample)
            ret.update(text)
            videos_tensor = self.get_video(sample, start, end, duration)
            # print(imgs_tensor.size())
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
            # except Exception as e:
            #     # print(f"Error while read file idx {sample} in {self.names[0]} -> {e}")
            #     index = random.randint(0, len(self.metadata) - 1)
            if try_time > max_try:
                print(f"Exceed max time Error while read file idx {sample} in {self.names[0]}")
        return ret

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, index):
        return self.get_suite(index)
