from .video_base_dataset import BaseDataset
import torch as th
import os
import numpy as np
import random
import ffmpeg
import json
from transforms.video.videoaug import VideoTransform
import subprocess

# {'timestamp_sec': 221.29666, 'narration_text': '#C C walks on the ground'}


class Ego4DDataset(BaseDataset):
    """EGO4D Video-Text loader."""

    def __init__(self, *args, split="", **kwargs):
    # def __init__(
    #         self,
    #         csv,
    #         video_root='',
    #         caption_root='',
    #         min_time=4.0,
    #         fps=16,
    #         num_frames=16,
    #         size=224,
    #         crop_only=False,
    #         center_crop=True,
    #         benchmark=False,
    #         token_to_word_path='data/dict.npy',
    #         max_words=20,
    #         num_candidates=1,
    #         random_left_right_flip=False,
    # ):
    #     """
    #     Args:
    #     """
        assert split in ["train", "val", "test"]
        self.split = split

        if split == "train":
            names = ["ego4d_train"]
        elif split == "val":
            names = ["ego4d_val"]
        elif split == "test":
            names = ["ego4d_test"]
        super().__init__(*args, **kwargs, names=names, text_column_name="caption")

        self._load_metadata()
        # for howto100
        self.min_time = 2.0
        self.size = 224
        self.fps = 5
        self.num_sec = self.video_num_frames / float(self.fps)
        self.crop_only = False
        self.center_crop = False
        self.benchmark = False
        self.num_candidates = 1
        self.random_flip = True
        # print(self.data_dir)
        # for howto caption dir
        self._load_metadata()
        # print(kwargs)
        # self.num_frames = kwargs['num_frames']
        self.video_transform = VideoTransform(mode=self.split, num_frames=self.num_frames)  # train or val model

    def _load_metadata(self):
        metadata_dir = './meta_data'
        split_files = {
            'train': 'ego4d/narration.json',
            'val': 'ego4d/narration.json',            # there is no test
            'test': 'ego4d/narration.json'
        }
        target_split_fp = split_files[self.split]
        with open(os.path.join(metadata_dir, target_split_fp), 'r') as jsonfile:
            metadata = json.load(jsonfile)
        # metadata = pd.read_csv(os.path.join(metadata_dir, target_split_fp), sep='\t')
        self.metadata = metadata
        self.meta_keys = list(metadata.keys())

    def __len__(self):
        return len(self.meta_keys)

    def get_video_len(self, video_path):
        duration = subprocess.check_output(
            ['ffprobe', '-i', video_path, '-show_entries', 'format=duration', '-v', 'quiet', '-of',
             'csv=%s' % ("p=0")])
        # print("ffmpeg duration is: {}".format(duration))
        duration = float(str(duration)[2:-3])  # b'1027.806000\n' -> 1027.806
        return duration

    def read_frames_ffmpeg(self, video_path, center, video_len):
        if center > video_len:
            center = video_len - 2 * self.num_sec
        start = int(max(0, center-self.min_time))
        end = int(min(video_len, center+self.min_time))
        start_seek = random.randint(start, int(max(start, end - self.num_sec)))
        # video is too short
        if video_len < 1:
            start_seek = 0
        if start_seek + self.num_sec + 0.1 > video_len:
            start_seek = video_len - self.num_sec - 0.1
        start_seek = max(start_seek, 0)
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
            cmd.output('pipe:', format='rawvideo', pix_fmt='rgb24')
            .run(capture_stdout=True, quiet=True)
        )
        video = np.frombuffer(out, np.uint8).reshape([-1, self.size, self.size, 3])
        video_tensor = th.from_numpy(np.copy(video))
        video_tensor = video_tensor.permute(3, 0, 1, 2) + 0.01
        if video_tensor.size()[1] != self.num_frames:
            print(video_tensor.size(), start, end, start_seek, video_len)
            # print("video length: {}".format(self.get_video_len_from_timestammp()))
        # add gausian noise here to prevent all blank boxez
        if video_tensor.shape[1] < self.num_frames:
            zeros = th.ones((3, self.num_frames - video_tensor.shape[1], self.size, self.size), dtype=th.uint8)
            video_tensor = th.cat((video_tensor, zeros), axis=1)
        return video_tensor[:, :self.num_frames]

    def _zero_pad_tensor_token(self, tensor, size):
        if len(tensor) >= size:
            return tensor[:size]
        else:
            zero = th.zeros(size - len(tensor)).long()
            return th.cat((tensor, zero), dim=0)

    def get_text(self, sample, index):
        text = sample['narration_text']
        # TODO: May need to be improved for edge cases.
        encoding = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_text_len,
            return_special_tokens_mask=True,
        )
        return {
            "text": (text, encoding),
            "img_index": index,
            "cap_index": index,
            "raw_index": index,
        }

    def get_false_text(self, rep):
        random_index = random.randint(0, len(self.metadata) - 1)
        # two annotations
        if random.random() < 0.5:
            meta = self.metadata[self.meta_keys[random_index]]['narration_pass_1']
        else:
            meta = self.metadata[self.meta_keys[random_index]]['narration_pass_2']
        sample = meta[random.randint(0, len(meta) - 1)]  # random choice one sample
        text = sample['narration_text']
        encoding = self.tokenizer(
            text,
            # padding="max_length",
            truncation=True,
            max_length=self.max_text_len,
            return_special_tokens_mask=True,
        )
        return {f"false_text_{rep}": (text, encoding)}

    def _get_video_path(self, sample):
        rel_video_fp = sample["video_path"] + '.mp4'
        full_video_fp = os.path.join(self.data_dir, rel_video_fp)
        if not os.path.exists(full_video_fp):
            Exception(IOError)
        return full_video_fp, rel_video_fp

    def get_raw_video(self, sample):
        abs_fp, rel_fp = self._get_video_path(sample)
        # in four seconds
        # print(sample)
        sample["video_len"] = self.get_video_len(abs_fp)
        center = sample['timestamp_sec']
        imgs = self.read_frames_ffmpeg(abs_fp, center, sample["video_len"]).permute(1, 0, 2, 3)
        # print(imgs.size())
        if imgs is None:
            raise Exception("Invalid video!", rel_fp)
        else:
            return imgs

    def get_video(self, sample):
        imgs_tensor = self.get_raw_video(sample)
        return imgs_tensor

    def get_false_video(self, rep):
        random_index = random.randint(0, len(self.metadata) - 1)
        # two annotations
        if random.random() < 0.5:
            meta = self.metadata[self.meta_keys[random_index]]['narration_pass_1']
        else:
            meta = self.metadata[self.meta_keys[random_index]]['narration_pass_2']
        if len(meta) < 1:
            return self.get_false_video(rep)
        sample = meta[random.randint(0, len(meta) - 1)]  # random choice one sample
        sample["video_path"] = self.meta_keys[random_index]  # video path
        imgs_tensor = self.get_raw_video(sample)
        return {f"false_image_{rep}": imgs_tensor}

    def get_suite(self, index):
        result = None
        while result is None:
            # two annotations
            try:
                if random.random() < 0.5:
                    meta = self.metadata[self.meta_keys[index]]['narration_pass_1']
                else:
                    meta = self.metadata[self.meta_keys[index]]['narration_pass_2']
                if len(meta) < 2:
                    random_index = random.randint(0, len(self.metadata) - 1)
                    return self.get_suite(random_index)
                sample = meta[random.randint(0, len(meta)-1)]  # random choice one sample
                sample["video_path"] = self.meta_keys[index]  # video path
                # print(sample)
                ret = dict()
                text = self.get_text(sample, index)
                ret.update({"replica": True if text["cap_index"] > 0 else False})
                ret.update(text)
                imgs_tensor = self.get_video(sample)
                # print(imgs_tensor.size())
                ret.update({
                    "image": imgs_tensor,
                    "img_index": index,
                    "cap_index": index,
                    "raw_index": index,
                })
                ret.update({"replica": True if ret["cap_index"] > 0 else False})
                for i in range(self.draw_false_image):
                    ret.update(self.get_false_video(i))
                for i in range(self.draw_false_text):
                    ret.update(self.get_false_text(i))
                result = True
            except Exception as e:
                # print(e)
                index = random.randint(0, len(self.metadata) - 1)
        return ret

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, index):
        return self.get_suite(index)
