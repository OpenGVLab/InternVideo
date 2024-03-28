from .video_base_dataset import BaseDataset, read_large_frames_decord
import pandas as pd
import os


# {'timestamp_sec': 221.29666, 'narration_text': '#C C walks on the ground'}


class Ego4DDataset(BaseDataset):
    """EGO4D Video-Text loader."""

    def __init__(self, *args, split="", **kwargs):
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

    def _load_metadata(self):
        metadata_dir = './meta_data/ego4d'
        split_files = {
            'train': 'ego4d_train_subset.csv',
            'val': 'ego4d_val_ts_clean.csv',
            'test': 'ego4d_val_ts_clean.csv' # there is no test
        }
        target_split_fp = split_files[self.split]
        self.metadata = pd.read_csv(os.path.join(metadata_dir, target_split_fp), sep='\t',  header=None, error_bad_lines=False)

    def _get_video_path(self, sample):
        rel_video_fp = sample[0] + '.mp4'
        full_video_fp = os.path.join(self.data_dir, 'videos', rel_video_fp)
        if not os.path.exists(full_video_fp):
            Exception(IOError)
        return full_video_fp, rel_video_fp

    def _get_caption(self, sample):
        return sample[6]

    def get_raw_video(self, sample):
        abs_fp, rel_fp = self._get_video_path(sample)
        # if int(sample[2]) > 600:
        #     raise Exception("Video is longer than 10m!", rel_fp)
        frame_end, frame_loc = int(sample[3]), int(sample[5])
        # imgs = video_reader(abs_fp, frame_loc, frame_end, self.num_frames)
        imgs = read_large_frames_decord(abs_fp, frame_loc, frame_end, self.num_frames)
        if imgs is None:
            raise Exception("Invalid video!", rel_fp)
        else:
            return imgs

