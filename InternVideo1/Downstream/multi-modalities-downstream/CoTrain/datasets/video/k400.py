import numpy as np
from .video_base_dataset import BaseDataset
import os
import random
from CoTrain.transforms.video.videoaug import VideoTransform


class K400Dataset(BaseDataset):
    def __init__(self, *args, split="", **kwargs):
        assert split in ["train", "val", "test"]
        self.split = split
        self.metadata = None
        self.ans_lab_dict = dict()
        if split == "train":
            names = ["k400_train"]
        elif split == "val":
            names = ["k400_val"]
        elif split == "test":
            names = ["k400_test"]
        super().__init__(
            *args,
            **kwargs,
            names=names,
            text_column_name="questions",
            remove_duplicate=False,
        )
        self.video_transform = VideoTransform(mode=self.split)  # train or val model
        self._load_metadata()

    def _load_metadata(self):
        metadata_dir = './meta_data/k400'
        split_files = {
            'train': 'k400_train_tsm.list',
            'val': 'k400_test_tsm.list',
            'test': 'k400_test_tsm.list'
        }
        target_split_fp = split_files[self.split]
        with open(os.path.join(metadata_dir, target_split_fp)) as f:
            self.metadata = f.readlines()
        answer_fp = os.path.join(metadata_dir, 'kinetics_label_map.txt')
        count = 0
        with open(answer_fp, 'r') as f:
            lines = f.readlines()
            for line in lines:
                self.ans_lab_dict[str(line.strip())] = count
                count += 1

    def _get_video_path(self, sample):
        # find the name is os.listdir() e.g. abseiling/0wR5jVB-WPk.mp4
        # /data/algceph/arcdata/Kinetics-400/train_zips/snowboarding/MCgJO4s1qBA_000129_000139.zip
        # -> snowboarding/MCgJO4s1qBA_000129_000139.mp4
        if self.split == 'train':
            rel_path = sample[0][46:-4] + '.mp4'
        else:
            # val maybe mkv. webm etc.
            fake_path = sample[0][44:-4]
            sub_dir, video_name = fake_path.split('/')
            rel_path = sub_dir
            for video in os.listdir(os.path.join(self.data_dir, self.split, sub_dir)):
                if video_name in video:
                    rel_path = os.path.join(rel_path, video)
                    break
        full_path = os.path.join(self.data_dir, self.split, rel_path)
        # print(full_path)
        return full_path, rel_path

    def get_text(self, sample):
        text = "A persion is doing [MASK]"
        encoding = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_text_len,
            return_special_tokens_mask=True,
        )
        return (text, encoding)

    def get_answer_label(self, sample):
        text = "None"
        # print(len(self.ans_lab_dict))
        ans_total_len = len(self.ans_lab_dict) + 1  # one additional class
        ans_label = int(sample[1])
        scores = np.zeros(ans_total_len).astype(int)
        scores[ans_label] = 1
        return text, ans_label, scores

    def __getitem__(self, index):
        result = None
        while result is None:
            sample = self.metadata[index].split('\t')
            try:
                video_tensor = self.get_video(sample)
                text = self.get_text(sample)
                qid = index
                if self.split != "test":
                    answers, labels, scores = self.get_answer_label(sample)
                else:
                    answers = list()
                    labels = list()
                    scores = list()
                result = True
            except Exception as e:
                print(f"Error while read file idx {sample[0]} -> {e}")
                index = random.randint(0, len(self.metadata) - 1)
        return {
            "video": video_tensor,
            "text": text,
            "vqa_answer": answers,
            "vqa_labels": labels,
            "vqa_scores": scores,
            "qid": qid,
        }

    def __len__(self):
        return len(self.metadata)