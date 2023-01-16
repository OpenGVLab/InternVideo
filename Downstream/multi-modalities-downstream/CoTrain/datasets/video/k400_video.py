from docutils import DataError
from importlib_metadata import metadata
from .video_base_dataset import BaseDataset, read_frames_decord
import os
import pandas as pd


class K400VideoDataset(BaseDataset):
    def __init__(self, *args, split="", **kwargs):
        assert split in ["train", "val", "test"]
        self.split = split
        if self.split == "train":
            Exception("no train data provided")
        self.metadata = None
        self.ans_texts = dict()
        if split == "train":
            names = ["k400_video_train"]
        elif split == "val":
            names = ["k400_video_val"]
        elif split == "test":
            names = ["k400_video_test"]  # vqav2_test-dev for test-dev

        super().__init__(
            *args,
            **kwargs,
            names=names,
            text_column_name="unknown",
            remove_duplicate=False,
        )
        self._load_metadata()
        self.data_dir = "s3://video_pub/K400_videos"  # TODO: Remove this piece of shit

    def _load_metadata(self):
        metadata_dir = './meta_data/k400'
        split_files = {
            'train': 'k400_test_tsm.list',
            'val': 'k400_test_tsm.list',
            'test': 'k400_test_tsm.list',
        }
        target_split_fp = split_files[self.split]
        with open(os.path.join(metadata_dir, target_split_fp)) as f:
            self.metadata = f.readlines()
        self.metadata = [x.strip() for x in self.metadata]
        self.metadata = [x.split("\t") for x in self.metadata]
        self.metadata = [[x[0].split("/")[-1][:11], int(x[1])] for x in self.metadata]

    def _build_ans(self):
        metadata_dir = './meta_data/k400'
        answer_fp = os.path.join(metadata_dir, 'kinetics_label_map.txt')
        ans_texts = open(answer_fp).readlines()
        assert len(set(ans_texts)) == 400
        ans_texts = [x.strip() for x in ans_texts]
        ans_texts = ["A person is doing {}".format(x) for x in ans_texts]
        ans_texts = [
            (
                x,
                self.tokenizer(
                    x,
                    padding="max_length",
                    truncation=True,
                    max_length=self.max_text_len,
                    return_special_tokens_mask=True,
                ),
            )
            for x in ans_texts
        ]
        self.ans_texts = {"text": ans_texts[0]}
        self.ans_texts.update(
            {
                "false_text_{}".format(i): ans_texts[i + 1]
                for i in range(len(ans_texts) - 1)
            }
        )

    @staticmethod
    def classes():
        metadata_dir = './meta_data/k400'
        answer_fp = os.path.join(metadata_dir, 'kinetics_label_map.txt')
        ans_texts = open(answer_fp).readlines()
        assert len(set(ans_texts)) == 400
        ans_texts = [x.strip() for x in ans_texts]
        return ans_texts

    def _get_video_path(self, sample):
        rel_video_fp = sample[0] + '.mp4'
        if "s3://" in self.data_dir:
            full_video_fp = os.path.join(self.data_dir, rel_video_fp)
        else:
            full_video_fp = os.path.join(self.data_dir, self.split, rel_video_fp)
        return full_video_fp, rel_video_fp

    def get_raw_video(self, sample):
        abs_fp, rel_fp = self._get_video_path(sample)
        videos, idxs, vlen = read_frames_decord(abs_fp, self.num_frames, mode=self.split)
        if videos is None:
            raise Exception("Invalid img!", rel_fp)
        else:
            return videos

    def get_video(self, sample):
        videos = self.get_raw_video(sample)
        videos_tensor = self.video_aug(videos, self.video_transform)
        return videos_tensor

    def __getitem__(self, index):
        ret = None
        max_try = 10
        try_time = 0
        while ret is None:
            try_time += 1
            try:
                sample = self.metadata[index]
                image_tensor = self.get_video(sample)
                answer = sample[1]
                ret = {
                    "video": image_tensor,
                    "img_index": index,
                    'answer': answer,
                }
            except Exception as e:
                index = (index + 1) % len(self.metadata)
                exc = e
            if try_time > max_try:
                raise DataError(
                    f"Exceed max time Error while read file idx {sample[0]} with error {exc}"
                )
        return ret

    def __len__(self):
        return len(self.metadata)