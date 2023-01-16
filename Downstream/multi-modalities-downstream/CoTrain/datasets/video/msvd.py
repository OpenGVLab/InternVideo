from .video_base_dataset import BaseDataset
import random
import os
import pandas as pd


class MSVDDataset(BaseDataset):
    def __init__(self, *args, split="", **kwargs):
        assert split in ["train", "val", "test"]
        self.split = split
        self.metadata = None
        if split == "train":
            names = ["msvd_train"]
        elif split == "val":
            names = ["msvd_val"]
        elif split == "test":
            names = ["msvd_test"]
        self._load_metadata()
        # self.num_frames = kwargs['num_frames']
        super().__init__(*args, **kwargs, names=names, text_column_name="caption")

    def _load_metadata(self):
        metadata_dir = './meta_data/msvd'
        split_files = {
            'train': 'MSVD_train.tsv',
            'val': 'MSVD_test.tsv',  # MSVD_val.tsv
            'test': 'MSVD_test.tsv'
        }
        target_split_fp = split_files[self.split]
        metadata = pd.read_csv(os.path.join(metadata_dir, target_split_fp), sep='\t')
        self.metadata = metadata
        print("load split {}, {} samples".format(self.split, len(metadata)))

    def _get_video_path(self, sample):
        rel_video_fp = sample[1] + '.avi'
        full_video_fp = os.path.join(self.data_dir, 'YouTubeClips', rel_video_fp)
        return full_video_fp, rel_video_fp

    def _get_caption(self, sample):
        if self.split == 'train':
            words = sample[0].split(',')
            num_word = len(words)
            index = random.randint(0, num_word - 1)
            caption = words[index]
        else:
            # caption = sample[0]
            words = sample[0].split(',')
            num_word = len(words)
            index = random.randint(0, num_word - 1)
            caption = words[index]
        return caption
