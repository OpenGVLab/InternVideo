from .video_base_dataset import BaseDataset, read_frames_decord
import random
import os
import pandas as pd


class WEBVIDDataset(BaseDataset):
    def __init__(self, *args, split="", **kwargs):
        assert split in ["train", "val", "test"]
        self.split = split
        self.metadata = None
        self.cut = "jsfusion"
        if split == "train":
            names = ["webvid_train"]
        elif split == "val":
            names = ["webvid_val"]
        elif split == "test":
            names = ["webvid_val"]
        self._load_metadata()

        print(names, ": ", len(self.metadata), "samples in total.")
        super().__init__(*args, **kwargs, names=names, text_column_name="caption")
        if "s3://" in self.data_dir:
            # Remove this fucking auto dir name
            self.data_dir = os.path.dirname(self.data_dir)
            # Add the real path
            self.data_dir = os.path.join(self.data_dir, "WebVid2M")

    def _load_metadata(self):
        metadata_dir = './meta_data/webvid'
        split_files = {
            'train': 'webvid_training_success_full.tsv',
            'val': 'webvid_validation_success_full.tsv',            # there is no test
            'test': 'webvid_validation_success_full.tsv'
        }
        target_split_fp = split_files[self.split]
        metadata = pd.read_csv(os.path.join(metadata_dir, target_split_fp), sep='\t')
        self.metadata = metadata

    def _get_video_path(self, sample):
        rel_video_fp = sample[1] + '.mp4'
        if "s3://" in self.data_dir:
            full_video_fp = os.path.join(self.data_dir, rel_video_fp)
        else:
            full_video_fp = os.path.join(self.data_dir, self.split, rel_video_fp)
        return full_video_fp, rel_video_fp

    def _get_caption(self, sample):
        return sample[0]

    def get_raw_video(self, sample):
        abs_fp, rel_fp = self._get_video_path(sample)
        videos, idxs, vlen = read_frames_decord(abs_fp, self.num_frames, mode=self.split)
        if videos is None:
            raise Exception("Invalid video!", rel_fp)
        else:
            return videos

    def get_video(self, index, sample):
        videos = self.get_raw_video(sample)
        videos_tensor = self.video_aug(videos, self.video_transform)
        return {
            "video": videos_tensor,
            "vid_index": index,
            "cap_index": index,
            "raw_index": index,
        }

    def get_false_video(self, rep):
        random_index = random.randint(0, len(self.metadata) - 1)
        sample = self.metadata.iloc[random_index]
        # can be different augmentation
        videos = self.get_raw_video(sample)
        videos_tensor = self.video_aug(videos, self.video_transform)
        return {f"false_video_{rep}": videos_tensor}

    def get_text(self, raw_index, sample):
        text = sample[0]
        # print(text)
        encoding = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_text_len,
            return_special_tokens_mask=True,
        )
        # print(encoding.size())
        return {
            "text": (text, encoding),
            "vid_index": raw_index,
            "cap_index": raw_index,
            "raw_index": raw_index,
        }

    def get_false_text(self, rep):
        random_index = random.randint(0, len(self.metadata) - 1)
        sample = self.metadata.iloc[random_index]
        text = sample[0]
        encoding = self.tokenizer(
            text,
            # padding="max_length",
            truncation=True,
            max_length=self.max_text_len,
            return_special_tokens_mask=True,
        )
        return {f"false_text_{rep}": (text, encoding)}

    def get_suite(self, index):
        result = None
        max_try = 10
        try_time = 0
        while result is None:
            try_time += 1
            sample = self.metadata.iloc[index]
            try:
                ret = dict()
                ret.update(self.get_video(index, sample))
                if not self.video_only:
                    txt = self.get_text(index, sample)
                    ret.update({"replica": True if txt["cap_index"] > 0 else False})
                    ret.update(txt)

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
                try_time=0
        return ret

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, index):
        return self.get_suite(index)