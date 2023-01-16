from .video_base_dataset import BaseDataset, read_large_frames_decord, get_video_len
import os
import pandas as pd


class EGO4DChoiceDataset(BaseDataset):
    def __init__(self, *args, split="", **kwargs):
        assert split in ["train", "val", "test"]
        self.split = split
        if self.split == "train":
            Exception("no train data provided")
        self.metadata = None
        self.ans_lab_dict = None
        if split == "train":
            names = ["ego4d_choice_train"]
        elif split == "val":
            names = ["ego4d_choice_val"]
        elif split == "test":
            names = ["ego4d_choice_test"]  # vqav2_test-dev for test-dev

        super().__init__(
            *args,
            **kwargs,
            names=names,
            text_column_name="unknown",
            remove_duplicate=False,
        )
        self._load_metadata()

    def _load_metadata(self):
        metadata_dir = './meta_data/ego4d'
        split_files = {
            'train': 'mc_val.csv',         # no train and test available, only for zero-shot testing
            'val': 'mc_val.csv',
            'test': 'mc_val.csv'
        }
        target_split_fp = split_files[self.split]
        self.metadata = pd.read_csv(os.path.join(metadata_dir, target_split_fp), sep=',', header=0, on_bad_lines='skip')

    def _get_video_path(self, sample):
        rel_video_fp = eval(sample["question"])[0] + '.mp4'
        full_video_fp = os.path.join(self.data_dir, 'videos', rel_video_fp)
        if not os.path.exists(full_video_fp):
            Exception(IOError)
        return full_video_fp, rel_video_fp

    def get_raw_video(self, sample):
        abs_fp, rel_fp = self._get_video_path(sample)
        frame_loc = eval(sample["question"])[1]
        frame_end = get_video_len(abs_fp)
        imgs = read_large_frames_decord(abs_fp, frame_loc, frame_end, self.num_frames, mode=self.split)
        if imgs is None:
            raise Exception("Invalid video!", rel_fp)
        else:
            return imgs

    def get_text(self, sample):
        texts = []
        for answer in eval(sample["answers"]):
            text = answer[-1]
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
        gt_text = eval(sample["question"])[-1]
        answer_label = 0
        for index, answer in enumerate(eval(sample["answers"])):
            if answer[-1] == gt_text:
                answer_label = index
        return answer_label

    def __getitem__(self, index):
        sample = self.metadata.iloc[index]
        # print(sample)
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
        return ret

    def __len__(self):
        return len(self.metadata)