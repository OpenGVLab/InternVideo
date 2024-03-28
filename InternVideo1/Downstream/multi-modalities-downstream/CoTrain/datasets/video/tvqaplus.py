from .video_base_dataset import BaseDataset
import os
import pandas as pd
import cv2
import torch
from CoTrain.datasets.video.video_base_dataset import sample_frames

# each sample: https://tvqa.cs.unc.edu/download_tvqa_plus.html
# {
#   "answer_idx": "1",
#   "qid": 134094,
#   "ts": [5.99, 11.98],
#   "a1": "Howard is talking to Raj and Leonard",
#   "a0": "Howard is talking to Bernadette",
#   "a3": "Howard is talking to Leonard and Penny",
#   "a2": "Howard is talking to Sheldon , and Raj",
#   "q": "Who is Howard talking to when he is in the lab room ?",
#   "vid_name": "s05e02_seg02_clip_00",
#   "a4": "Howard is talking to Penny and Bernadette",
#   "bbox": {
#     "14": [
#       {
#         "img_id": 14,
#         "top": 153,
#         "label": "Howard",
#         "width": 180,
#         "height": 207,
#         "left": 339
#       },
#       {
#         "img_id": 14,
#         "top": 6,
#         "label": "lab",
#         "width": 637,
#         "height": 354,
#         "left": 3
#       },
#       ...
#     ],
#     "20": [ ... ],
#     "26": [ ... ],
#     "32": [ ... ],
#     "38": [ ... ]
#   }
# }


class TVQAPLUSDataset(BaseDataset):
    def __init__(self, *args, split="", **kwargs):
        assert split in ["train", "val", "test"]
        self.split = split
        self.metadata = None
        self._load_metadata()
        if split == "train":
            names = ["tvqaplus_train"]
        elif split == "val":
            names = ["tvqaplus_val"]
        elif split == "test":
            names = ["tvqaplus_test"]

        super().__init__(*args, **kwargs, names=names, text_column_name="caption")
        # for appear objects
        self.only_use_relevant_dets = True
        if self.only_use_relevant_dets:
            self.relevant_dets = []  # resort the detection numbers
            self.relevant_dets_classes = []

    def _load_metadata(self):
        # download specific
        metadata_dir = './meta_data/tvqa'
        split_files = {
            'train': 'tvqa_plus_train.jsonl',
            'val': 'tvqa_plus_val.jsonl',
            'test': 'tvqa_plus_test_public.jsonl'  # no GT label for test set
        }
        target_split_fp = split_files[self.split]
        metadata = pd.read_json(os.path.join(metadata_dir, target_split_fp), lines=True)
        self.metadata = metadata

    def _get_image_path(self, sample):
        rel_fp = sample['vid_name']
        return os.path.join(self.data_dir, rel_fp), rel_fp

    def _get_caption(self, sample):
        return sample[0]

    # tvqaplus provide sampled frames (3 fps)
    # To Do: considering sample one frame with bounding box
    def get_raw_video(self, sample):
        abs_fp, rel_fp = self._get_image_path(sample)
        [beg_time, end_time] = sample['ts']
        clip_len = int((float(end_time) - float(beg_time)) * 3)
        rel_frame_index = sample_frames(self.num_frames, clip_len)
        # sample N frames here
        frames = []
        for index in rel_frame_index:
            img = cv2.imread(abs_fp + '{}.jpg'.format(index))
            frame = torch.from_numpy(img).byte()
            frame = frame.permute(2, 0, 1)
            frames.append(frame)
        frames = torch.stack(frames).permute(1, 0, 2, 3)
        return frames

    def get_text(self, sample):
        question = self.get_question(sample)
        qa_texts = []
        # 5 choices
        for i in range(5):
            raw_text = question + "[SEP]" + sample["a{}".format(i)]
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
        answer = int(sample['answer_idx'])
        return answer

    def get_question(self, sample):
        return sample["q"]

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, index):
        sample = self.metadata.iloc[index]
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
        ret["video"] = video_tensor
        return ret

