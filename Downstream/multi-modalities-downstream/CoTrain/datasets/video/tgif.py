import numpy as np
from .video_base_dataset import BaseDataset, read_frames_gif
import os
import json
import pandas as pd
import random

# 2022.1.28 read gif is too slow, may be need to speedup by convert gif -> video
# https://stackify.dev/833655-python-convert-gif-to-videomp4


class TGIFDataset(BaseDataset):
    def __init__(self, *args, split="", **kwargs):
        assert split in ["train", "val", "test"]
        self.split = split
        self.metadata = None
        self.ans_lab_dict = None
        if split == "train":
            names = ["tgif_train"]
        elif split == "val":
            names = ["tgif_val"]
        elif split == "test":
            names = ["tgif_test"]

        super().__init__(
            *args,
            **kwargs,
            names=names,
            text_column_name="questions",
            remove_duplicate=False,
        )
        # self.num_frames = 4
        self._load_metadata()
        self.data_dir = "/mnt/lustre/share_data/heyinan/data/tgif"  # TODO: Remove this piece of shit

    def _load_metadata(self):
        metadata_dir = './meta_data/tgif'
        split_files = {
            'train': 'frameqa_train.jsonl',
            'val': 'frameqa_test.jsonl',  # frameqa_val.jsonl
            'test': 'frameqa_test.jsonl'
        }
        target_split_fp = split_files[self.split]
        answer_fp = os.path.join(metadata_dir, 'frameqa_trainval_ans2label.json')
        # answer_fp = os.path.join(metadata_dir, 'msrvtt_qa_ans2label.json')
        with open(answer_fp, 'r') as JSON:
            self.ans_lab_dict = json.load(JSON)
        # path_or_buf=os.path.join(metadata_dir, target_split_fp)
        metadata = pd.read_json(os.path.join(metadata_dir, target_split_fp), lines=True)
        self.metadata = metadata

    def _get_video_path(self, sample):
        return os.path.join(self.data_dir, 'gifs', sample['gif_name']) + '.gif', sample['gif_name'] + '.gif'

    def get_raw_video(self, sample):
        abs_fp, rel_fp = self._get_video_path(sample)
        imgs, idxs, vlen = read_frames_gif(abs_fp, self.num_frames, mode=self.split)
        if imgs is None:
            raise Exception("Invalid img!", rel_fp)
        else:
            return imgs

    def get_text(self, sample):
        text = sample['question']
        encoding = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_text_len,
            return_special_tokens_mask=True,
        )
        return (text, encoding)

    def get_answer_label(self, sample):
        text = sample['answer']
        ans_total_len = len(self.ans_lab_dict) + 1  # one additional class
        try:
            ans_label = self.ans_lab_dict[text]  #
        except KeyError:
            ans_label = -100  # ignore classes
            # ans_label = 1500 # other classes
        scores = np.zeros(ans_total_len).astype(int)
        scores[ans_label] = 1
        return text, ans_label, scores
        # return text, ans_label_vector, scores

    def __getitem__(self, index):
        result = None
        while result is None:
            sample = self.metadata.iloc[index]
            try:
                video_tensor = self.get_video(sample)
                text = self.get_text(sample)
                # index, question_index = self.index_mapper[index]
                qid = index
                result = True
            except Exception as e:
                gif_name = sample["gif_name"]
                print(f"Error while read file idx {gif_name}")
                assert self.split != "test"
                index = random.randint(0, len(self.metadata) - 1)

        if self.split != "test":
            answers, labels, scores = self.get_answer_label(sample)
        else:
            answers = list()
            labels = list()
            scores = list()

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