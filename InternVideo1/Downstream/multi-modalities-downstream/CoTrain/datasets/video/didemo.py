from .video_base_dataset import BaseDataset
import os
import pandas as pd

# some videos are missed, for better results, do IO exception.


class DIDEMODataset(BaseDataset):
    def __init__(self, *args, split="", **kwargs):
        assert split in ["train", "val", "test"]
        self.split = split
        self.metadata = None
        if split == "train":
            names = ["didemo_train"]
        elif split == "val":
            names = ["didemo_val"]
        elif split == "test":
            names = ["didemo_val"]
        super().__init__(*args, **kwargs, names=names, text_column_name="caption")
        self._load_metadata()

    def _load_metadata(self):
        metadata_dir = './meta_data/didemo'
        split_files = {
            'train': 'DiDeMo_train.tsv',
            'val': 'DiDeMo_val.tsv',  # there is no test
            'test': 'DiDeMo_test.tsv'
        }
        target_split_fp = split_files[self.split]
        metadata = pd.read_csv(os.path.join(metadata_dir, target_split_fp), sep='\t')
        self.metadata = metadata
        print("load split {}, {} samples".format(self.split, len(metadata)))

    def _get_video_path(self, sample):
        rel_video_fp = sample[1]
        full_video_fp = os.path.join(self.data_dir, '', rel_video_fp)
        return full_video_fp, rel_video_fp

    def _get_caption(self, sample):
        return sample[0]

