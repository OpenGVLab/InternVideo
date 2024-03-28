import numpy as np
from .video_base_dataset import BaseDataset
import os
import json
import pandas as pd


class MSVDQADataset(BaseDataset):
    def __init__(self, *args, split="", **kwargs):
        assert split in ["train", "val", "test"]
        self.split = split
        self.metadata = None
        self.ans_lab_dict = None
        if split == "train":
            names = ["msvd_qa_train"]
        elif split == "val":
            names = ["msvd_qa_test"]  # test: directly output test result
            # ["msvd_qa_val"]
        elif split == "test":
            names = ["msvd_qa_test"]  # vqav2_test-dev for test-dev

        super().__init__(
            *args,
            **kwargs,
            names=names,
            text_column_name="questions",
            remove_duplicate=False,
        )
        self._load_metadata()
        self.data_dir = "s3://video_pub/MSVD/"  # TODO: Remove this piece of shit

    def _load_metadata(self):
        metadata_dir = './meta_data/msvd'
        split_files = {
            'train': 'msvd_train_qa_encode.json',
            'val': 'msvd_val_qa_encode.json',
            'test': 'msvd_test_qa_encode.json'
        }
        # read ans dict
        self.ans_lab_dict = {}
        answer_fp = os.path.join(metadata_dir, 'msvd_answer_set.txt')
        answer_clip_id = os.path.join(metadata_dir, 'msvd_clip_id.json')
        self.youtube_mapping_dict = dict()
        with open(os.path.join(metadata_dir, 'msvd_youtube_mapping.txt')) as f:
            lines = f.readlines()
            for line in lines:
                info = line.strip().split(' ')
                self.youtube_mapping_dict[info[1]] = info[0]
        with open(answer_fp, 'r') as f:
            lines = f.readlines()
            count = 0
            for line in lines:
                self.ans_lab_dict[str(line.strip())] = count
                count += 1
        with open(answer_clip_id, 'r') as JSON:
            self.ans_clip_id = json.load(JSON)
        for name in self.names:
            split = name.split('_')[-1]
            target_split_fp = split_files[split]
            metadata = pd.read_json(os.path.join(metadata_dir, target_split_fp), lines=True)
            if self.metadata is None:
                self.metadata = metadata
            else:
                self.metadata.update(metadata)
        print("total {} samples for {}".format(sum(1 for line in self.metadata), self.names))

    def _get_video_path(self, sample):
        rel_video_fp = self.youtube_mapping_dict['vid' + str(sample["video_id"])] + '.avi'
        # print(rel_video_fp)
        full_video_fp = os.path.join(self.data_dir, 'MSVD_Videos', rel_video_fp)
        return full_video_fp, rel_video_fp

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
        sample = self.metadata[index].iloc[0]
        video_tensor = self.get_video(sample)
        text = self.get_text(sample)
        # index, question_index = self.index_mapper[index]
        qid = index
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
            "ans_clip_id": self.ans_clip_id,
        }

    def __len__(self):
        return sum(1 for line in self.metadata)  # count # json lines