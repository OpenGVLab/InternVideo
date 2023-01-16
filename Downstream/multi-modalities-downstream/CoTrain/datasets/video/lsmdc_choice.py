from .video_base_dataset import BaseDataset
import os
import pandas as pd
import random
from .pack_meta import pack_metadata, unpack_metadata


class LSMDCChoiceDataset(BaseDataset):
    def __init__(self, *args, split="", **kwargs):
        assert split in ["train", "val", "test"]
        self.split = split
        self.metadata = None
        self.ans_lab_dict = None
        if split == "train":
            names = ["lsmdc_choice_train"]
        elif split == "val":
            names = ["lsmdc_choice_val"]
        elif split == "test":
            names = ["lsmdc_choice_test"]  # vqav2_test-dev for test-dev

        super().__init__(
            *args,
            **kwargs,
            names=names,
            text_column_name="unknown",
            remove_duplicate=False,
        )
        self._load_metadata()
        if "s3://" in self.data_dir:
            # Remove this fucking auto dir name
            self.data_dir = os.path.dirname(self.data_dir)
            # Add the real path
            self.data_dir = os.path.join(self.data_dir, "LSMDC")

    def _load_metadata(self):
        metadata_dir = './meta_data/lsmdc'
        split_files = {
            'train': 'LSMDC16_multiple_choice_train.csv',
            'val': 'LSMDC16_multiple_choice_test_randomized.csv',  # 'LSMDC16_multiple_choice_valid.csv',
            'test': 'LSMDC16_multiple_choice_test_randomized.csv'
        }
        target_split_fp = split_files[self.split]
        print(os.path.join(metadata_dir, target_split_fp))
        metadata = pd.read_csv(os.path.join(metadata_dir, target_split_fp), sep='\t', header=None, on_bad_lines='skip')
        datalist = []
        data_to_ignore = [
            "3056_PUBLIC_ENEMIES_01.24.29.351-01.24.32.274",
            "3035_INSIDE_MAN_02.02.18.839-02.02.25.201",
            "3064_SPARKLE_2012_00.14.12.000-00.14.22.429",
        ]
        for raw_id in range(len(metadata)):
            raw_d = metadata.iloc[raw_id]
            video_fp = raw_d[0]  # 3001_21_JUMP_STREET_00.03.07.077-00.03.07.559
            if video_fp.strip() in data_to_ignore:
                continue
            sub_path = video_fp.split('.')[0]  # 3001_21_JUMP_STREET_00
            remove = sub_path.split('_')[-1]  # 00
            sub_path = sub_path.replace('_'+remove,'/')  # 3001_21_JUMP_STREET/
            rel_video_fp = sub_path + video_fp + '.avi' # 
            options = [raw_d[idx] for idx in range(5, 10)]
            d = dict(
                id=video_fp,
                vid_id=rel_video_fp,
                answer=raw_d[10] - 1 if self.split in ['val', 'test'] else 0,
                options=options,
            )
            datalist.append(d)
        self.metadata = pack_metadata(self, pd.DataFrame(datalist))
        print("load split {}, {} samples".format(self.split, len(self.metadata)))

    def _get_video_path(self, sample):
        rel_video_fp = sample['vid_id']
        full_video_fp = os.path.join(self.data_dir, rel_video_fp)
        # print(full_video_fp)
        # assert os.path.exists(full_video_fp)
        return full_video_fp, rel_video_fp

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
        result = False
        while not result:
            try:
                sample = unpack_metadata(self, index)
                video_tensor = self.get_video(sample)
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
                for i in range(self.draw_false_text - 1):
                    ret.update({f"false_text_{i}": texts[i+1]})
                result = True
            except Exception as e:
                print(f"Error while read file idx {sample['vid_id']} in {self.names[0]} -> {e}")
                index = random.randint(0, len(self.metadata) - 1)
        return ret

    def __len__(self):
        return len(self.metadata)