from .video_base_dataset import BaseDataset
import random
import os
import pandas as pd
import cv2
import torch
from CoTrain.datasets.video.video_base_dataset import sample_frames

# each sample: https://tvqa.cs.unc.edu/download_tvqa.html
# {
# "a0": "A martini glass",
# "a1": "Nachos",
# "a2": "Her purse",
# "a3": "Marshall's book",
# "a4": "A beer bottle",
# "answer_idx": 4,
# "q": "What is Robin holding in her hand when she is talking to Ted about Zoey?",
# "qid": 7,
# "ts": "1.21-8.49",
# "vid_name": "met_s06e22_seg01_clip_02",
# "show_name":
# }


class TVQADataset(BaseDataset):
    def __init__(self, *args, split="", **kwargs):
        assert split in ["train", "val", "test"]
        self.split = split
        self.metadata = None
        self._load_metadata()
        if split == "train":
            names = ["tvqa_train"]
        elif split == "val":
            names = ["tvqa_val"]
        elif split == "test":
            names = ["tvqa_test"]

        super().__init__(*args, **kwargs, names=names, text_column_name="caption")
        # for appear objects
        self.only_use_relevant_dets = True
        if self.only_use_relevant_dets:
            self.relevant_dets = []  # resort the detection numbers
            self.relevant_dets_classes = []
        self.fps = 3  # tvqa sample 3 frames per second

    def _load_metadata(self):
        # download specific
        metadata_dir = './meta_data/tvqa'
        split_files = {
            'train': 'tvqa_train.jsonl',
            'val': 'tvqa_val.jsonl',
            'test': 'tvqa_test_public.jsonl'  # no GT label for test set
        }
        target_split_fp = split_files[self.split]
        metadata = pd.read_json(os.path.join(metadata_dir, target_split_fp), lines=True)
        self.metadata = metadata

    def _get_image_path(self, sample):
        # for example: tvqa/frames/raw_frames/frames_hq/met_frames/met_s06e22_seg01_clip_02
        dir_name = sample['vid_name'].split('_')[0]
        if dir_name not in ['bbt', 'castle', 'friends',  'grey',  'house', 'met']:
            dir_name = 'bbt'
        rel_fp = os.path.join('frames/raw_frames/frames_hq/', dir_name + '_frames', sample['vid_name'])
        return os.path.join(self.data_dir, rel_fp), rel_fp

    def _get_caption(self, sample):
        return sample[0]

    # need to speed up
    def _get_video_len(self, dir):
        return len(os.listdir(dir))

    # tvqa provide sampled frames
    def get_raw_video(self, sample):
        abs_fp, rel_fp = self._get_image_path(sample)
        [beg_time, end_time] = sample['ts'].split('-')
        clip_len = int((float(end_time) - float(beg_time)) * self.fps)
        # try:
        #     clip_len = int((float(end_time) - float(beg_time)) * self.fps)
        # except ValueError:
        #     clip_len = 1
        # prevent short than 1 second
        clip_len = max(clip_len, 2*self.num_frames)
        rel_frame_index = sample_frames(self.num_frames, clip_len)
        begin_frame_index = max(1, int(float(beg_time) * self.fps))
        video_len = self._get_video_len(abs_fp)
        # sample N frames here
        frames = []
        for index in rel_frame_index:
            abs_index = begin_frame_index + index
            abs_index = min(video_len, abs_index)
            image_rel_path = f'{abs_index:05}'
            img = cv2.imread(os.path.join(abs_fp, '{}.jpg'.format(image_rel_path)))
            # print(img)
            # print(os.path.join(abs_fp, '{}.jpg'.format(image_rel_path)))
            if img is None:
                print(sample['vid_name'])
                print(os.path.join(abs_fp, '{}.jpg'.format(image_rel_path)))
            frame = torch.from_numpy(img).byte()
            frame = frame.permute(2, 0, 1)
            frames.append(frame)
        frames = torch.stack(frames).permute(1, 0, 2, 3)
        return frames

    def get_text(self, sample):
        question = self.get_question(sample)
        qa_texts = []
        # 5 choices # ClipBERT: " ", Ours: [SEP]
        # if the length suppress than 40 ?
        options = " ".join(sample["a{}".format(i)] for i in range(5))
        for i in range(5):
            raw_text = question + "Options: " + options + "Answer: " + sample["a{}".format(i)]
            # raw_text = question + "[SEP]" + sample["a{}".format(i)]
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
        answer = int(sample['answer_idx'])
        return answer

    def get_question(self, sample):
        return sample["q"]

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

