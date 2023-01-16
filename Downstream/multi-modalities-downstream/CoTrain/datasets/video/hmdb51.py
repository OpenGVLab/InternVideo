import numpy as np
from .video_base_dataset import BaseDataset
import os


class HMDB51Dataset(BaseDataset):
    def __init__(self, *args, split="", **kwargs):
        assert split in ["train", "val", "test"]
        self.split = split
        self.metadata = None
        self.ans_lab_dict = dict()
        if split == "train":
            names = ["hmdb51_train"]
        elif split == "val":
            names = ["hmdb51_val"]
        elif split == "test":
            names = ["hmdb51_test"]
        super().__init__(
            *args,
            **kwargs,
            names=names,
            text_column_name="questions",
            remove_duplicate=False,
        )
        self._load_metadata()

    def _load_metadata(self):
        metadata_dir = './meta_data/hmdb51'
        split_files = {
            'train': 'hmdb51_rgb_train_split_1.txt',
            'val': 'hmdb51_rgb_val_split_1.txt',
            'test': 'hmdb51_rgb_val_split_1.txt'
        }
        target_split_fp = split_files[self.split]
        self.metadata = [x.strip().split(' ') for x in open(os.path.join(metadata_dir, target_split_fp))]
        answer_fp = os.path.join(metadata_dir, 'hmdb51_classInd.txt')
        with open(answer_fp, 'r') as f:
            lines = f.readlines()
            for line in lines:
                self.ans_lab_dict[str(int(line.strip().split(' ')[0]) - 1)] = line.strip().split(' ')[1]

    def _get_video_path(self, sample):
        # self.ans_lab_dict[sample[2]],
        return os.path.join(self.data_dir, sample[0].split('/')[-1]) + '.avi', sample[0].split('/')[-1] + '.avi'

    def get_text(self, sample):
        text = "A person is doing [MASK]"
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
        ans_total_len = len(self.ans_lab_dict) + 1  # one additional class
        ans_label = int(sample[2])
        scores = np.zeros(ans_total_len).astype(int)
        scores[ans_label] = 1
        return text, ans_label, scores
        # return text, ans_label_vector, scores

    def __getitem__(self, index):
        sample = self.metadata[index]  # .split(' ')
        video_tensor = self.get_video(sample)
        text = self.get_text(sample)
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
        }

    def __len__(self):
        return len(self.metadata)