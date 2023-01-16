from .video_base_dataset import BaseDataset
import os
import pandas as pd
from .pack_meta import pack_metadata, unpack_metadata


class MSRVTTChoiceDataset(BaseDataset):
    def __init__(self, *args, split="", **kwargs):
        assert split in ["train", "val", "test"]
        self.split = split
        if self.split == "train":
            Exception("no train data provided")
        self.metadata = None
        self.ans_lab_dict = None
        if split == "train":
            names = ["msrvtt_choice_train"]
        elif split == "val":
            names = ["msrvtt_choice_val"]
        elif split == "test":
            names = ["msrvtt_choice_test"]  # vqav2_test-dev for test-dev
        
        # Since the data is distribued like everywhere
        # We manully change data_dir
        args = ("./meta_data", *args[1:])

        super().__init__(
            *args,
            **kwargs,
            names=names,
            text_column_name="unknown",
            remove_duplicate=False,
        )
        self._load_metadata()

    def _load_metadata(self):
        metadata_dir = './meta_data/msrvtt'
        split_files = {
            'train': 'msrvtt_mc_test.jsonl',         # no train and test available, only for zero-shot
            'val': 'msrvtt_mc_test.jsonl',
            'test': 'msrvtt_mc_test.jsonl'
        }
        target_split_fp = split_files[self.split]
        metadata = pd.read_json(os.path.join(metadata_dir, target_split_fp), lines=True)
        self.metadata = pack_metadata(self, metadata)

    def _get_video_path(self, sample):
        return os.path.join(self.data_dir, 'videos', 'all', sample['clip_name'] + '.mp4'), sample['clip_name'] + '.mp4'

    def get_text(self, sample):
        texts = []
        for text in sample['options']:
            encoding = self.tokenizer(
                text,
                padding="max_length",
                truncation=True,
                max_length=self.max_text_len,
                return_special_tokens_mask=True,
            )
            texts.append((text, encoding))
        return texts

    def get_answer_label(self, sample):
        answer = sample['answer']
        return answer

    def __getitem__(self, index):
        sample = unpack_metadata(self, index)
        video_tensor = self.get_video(sample)
        # index, question_index = self.index_mapper[index]
        qid = index
        answer = self.get_answer_label(sample)
        ret = {
            "video": video_tensor,
            "vid_index": index,
            "cap_index": index,
            "raw_index": index,
            'answer': answer
        }
        texts = self.get_text(sample)
        ret["text"] = texts[0]
        # print(len(texts))
        for i in range(self.draw_false_text - 1):
            ret.update({f"false_text_{i}": texts[i+1]})
        # for i in range(self.draw_false_text-1):
        #     ret[f"false_text_{i}"] = texts[i+1]
        # print(ret.keys())
        return ret

    def __len__(self):
        return len(self.metadata)