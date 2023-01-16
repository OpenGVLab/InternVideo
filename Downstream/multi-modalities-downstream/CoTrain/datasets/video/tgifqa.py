from .video_base_dataset import BaseDataset, read_frames_gif
import random
import os
import pandas as pd

# action and transition: {
#     "gif_name": "tumblr_nk172bbdPI1u1lr18o1_250",
#     "question": "What does the butterfly do 10 or more than 10 times ?",
#     "options": ["stuff marshmallow", "holds a phone towards face",
#                 "fall over", "talk", "flap wings"],
#     "answer": 4
# }


class TGIFQADataset(BaseDataset):
    def __init__(self, *args, split="", **kwargs):
        assert split in ["train", "val", "test"]
        self.split = split
        self.data_split = "action"  # transition/action
        self.metadata = None
        self._load_metadata()
        if split == "train":
            names = ["tgifqa_train"]
        elif split == "val":
            names = ["tgifqa_val"]
        elif split == "test":
            names = ["tgifqa_test"]

        super().__init__(*args, **kwargs, names=names, text_column_name="caption")
        # for appear objects
        self.only_use_relevant_dets = True
        if self.only_use_relevant_dets:
            self.relevant_dets = []  # resort the detection numbers
            self.relevant_dets_classes = []
        self.fps = 3  # tgif sample 3 frames per second
        self.data_dir = "/mnt/lustre/share_data/heyinan/data/tgif"  # TODO: Remove this piece of shit

    def _load_metadata(self):
        # download specific
        metadata_dir = './meta_data/tgif'
        if self.data_split == "action":
            split_files = {
                'train': 'action_train.jsonl',
                'val': 'action_test.jsonl',  # action_val.jsonl
                'test': 'action_test.jsonl'  # no GT label for test set
            }
        elif self.data_split == "transition":
            split_files = {
                'train': 'transition_train.jsonl',
                'val': 'transition_test.jsonl',  # transition_val.jsonl
                'test': 'transition_test.jsonl'  # no GT label for test set
            }
        else:
            Exception("not support split")
        target_split_fp = split_files[self.split]
        metadata = pd.read_json(os.path.join(metadata_dir, target_split_fp), lines=True)
        self.metadata = metadata

    # def _get_image_path(self, sample):
    #     # for example: tvqa/frames/raw_frames/frames_hq/met_frames/met_s06e22_seg01_clip_02
    #     dir_name = sample['vid_name'].split('_')[0]
    #     if dir_name not in ['bbt', 'castle', 'friends',  'grey',  'house', 'met']:
    #         dir_name = 'bbt'
    #     rel_fp = os.path.join('frames/raw_frames/frames_hq/', dir_name + '_frames', sample['vid_name'])
    #     return os.path.join(self.data_dir, rel_fp), rel_fp

    def _get_caption(self, sample):
        return sample[0]

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
        question = self.get_question(sample)
        qa_texts = []
        # 5 choices # ClipBERT: " ", Ours: [SEP]
        options = " ".join(sample["options"][i] for i in range(5))
        for i in range(5):
            raw_text = question + "Options: " + options + "Answer: " + sample["options"][i]
            # raw_text = question + "[SEP]" + sample["options"][i]
            # print(raw_text)
            qa_encoding = self.tokenizer(
                raw_text,
                padding="max_length",
                truncation=True,
                max_length=self.max_text_len,
                return_special_tokens_mask=True,
            )
            qa_texts.append((raw_text, qa_encoding))
        return qa_texts

    def get_answer_label(self, sample):
        answer = int(sample['answer'])
        return answer

    def get_question(self, sample):
        return sample["question"]

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, index):
        result = None
        while result is None:
            sample = self.metadata.iloc[index]
            try:
                self.relevant_dets = []  # initalize
                self.relevant_dets_classes = []
                answer = self.get_answer_label(sample)
                ret = {
                    "vid_index": index,
                    "cap_index": index,
                    "raw_index": index,
                    'answer': answer
                }
                qa_texts = self.get_text(sample)
                ret["text"] = qa_texts[0]
                for i in range(self.draw_options_text - 1):
                    ret.update({f"options_text_{i}": qa_texts[i+1]})
                video_tensor = self.get_video(sample)
                ret["image"] = video_tensor
                result = True
            except Exception as e:
                print(f"Error while read file idx {sample.name} in {self.names[0]} -> {e}")
                print("time stamp is: {}".format(sample['ts']))
                index = random.randint(0, len(self.metadata) - 1)
        return ret

