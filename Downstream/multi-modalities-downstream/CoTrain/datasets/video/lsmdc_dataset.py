from .video_base_dataset import BaseDataset
import random
import os
import pandas as pd


class LSMDCDataset(BaseDataset):
    def __init__(self, *args, split="", **kwargs):
        assert split in ["train", "val", "test"]
        self.split = split
        self.metadata = None
        if split == "train":
            names = ["lsmdc_train"]
        elif split == "val":
            names = ["lsmdc_val"]
        elif split == "test":
            names = ["lsmdc_test"]
        self._load_metadata()
        # self.num_frames = kwargs['num_frames']
        super().__init__(*args, **kwargs, names=names, text_column_name="caption")

    def _load_metadata(self):
        metadata_dir = './meta_data/lsmdc'
        split_files = {
            'train': 'LSMDC16_annos_training.csv',
            'val': 'LSMDC16_challenge_1000_publictect.csv',  # LSMDC16_annos_val.csv
            'test': 'LSMDC16_challenge_1000_publictect.csv'
        }
        target_split_fp = split_files[self.split]
        metadata = pd.read_csv(os.path.join(metadata_dir, target_split_fp), sep='\t', header=None, error_bad_lines=False)
        self.metadata = metadata
        print("load split {}, {} samples".format(self.split, len(metadata)))

    def _get_video_path(self, sample):
        # e.g. 3009_BATTLE_LOS_ANGELES_00.03.07.170-00.03.09.675 -> 3009_BATTLE_LOS_ANGELES/3009_BATTLE_LOS_ANGELES_00.03.07.170-00.03.09.675
        sub_dir = '_'.join(sample[0].split('_')[:-1])
        rel_video_fp = sample[0] + '.avi'
        full_video_fp = os.path.join(self.data_dir, sub_dir, rel_video_fp)
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
