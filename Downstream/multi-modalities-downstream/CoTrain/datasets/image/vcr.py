from CoTrain.datasets.video.video_base_dataset import BaseDataset, color_img
import random
import os
import pandas as pd
import cv2
from CoTrain.transforms.video.videoaug import VideoTransform
import torch

## from https://github.com/rowanz/r2c/blob/master/dataloaders/vcr.py
# Here's an example jsonl
# {
# "movie": "3015_CHARLIE_ST_CLOUD",
# "objects": ["person", "person", "person", "car"],
# "interesting_scores": [0],
# "answer_likelihood": "possible",
# "img_fn": "lsmdc_3015_CHARLIE_ST_CLOUD/3015_CHARLIE_ST_CLOUD_00.23.57.935-00.24.00.783@0.jpg",
# "metadata_fn": "lsmdc_3015_CHARLIE_ST_CLOUD/3015_CHARLIE_ST_CLOUD_00.23.57.935-00.24.00.783@0.json",
# "answer_orig": "No she does not",
# "question_orig": "Does 3 feel comfortable?",
# "rationale_orig": "She is standing with her arms crossed and looks disturbed",
# "question": ["Does", [2], "feel", "comfortable", "?"],
# "answer_match_iter": [3, 0, 2, 1],
# "answer_sources": [3287, 0, 10184, 2260],
# "answer_choices": [
#     ["Yes", "because", "the", "person", "sitting", "next", "to", "her", "is", "smiling", "."],
#     ["No", "she", "does", "not", "."],
#     ["Yes", ",", "she", "is", "wearing", "something", "with", "thin", "straps", "."],
#     ["Yes", ",", "she", "is", "cold", "."]],
# "answer_label": 1,
# "rationale_choices": [
#     ["There", "is", "snow", "on", "the", "ground", ",", "and",
#         "she", "is", "wearing", "a", "coat", "and", "hate", "."],
#     ["She", "is", "standing", "with", "her", "arms", "crossed", "and", "looks", "disturbed", "."],
#     ["She", "is", "sitting", "very", "rigidly", "and", "tensely", "on", "the", "edge", "of", "the",
#         "bed", ".", "her", "posture", "is", "not", "relaxed", "and", "her", "face", "looks", "serious", "."],
#     [[2], "is", "laying", "in", "bed", "but", "not", "sleeping", ".",
#         "she", "looks", "sad", "and", "is", "curled", "into", "a", "ball", "."]],
# "rationale_sources": [1921, 0, 9750, 25743],
# "rationale_match_iter": [3, 0, 2, 1],
# "rationale_label": 1,
# "img_id": "train-0",
# "question_number": 0,
# "annot_id": "train-0",
# "match_fold": "train-0",
# "match_index": 0,
# }


class VCRDataset(BaseDataset):
    def __init__(self, *args, split="", **kwargs):
        assert split in ["train", "val", "test"]
        self.split = split
        self.metadata = None
        self._load_metadata()
        if split == "train":
            names = ["vcr_train"]
        elif split == "val":
            names = ["vcr_val"]
        elif split == "test":
            names = ["vcr_test"]

        super().__init__(*args, **kwargs, names=names, text_column_name="caption")
        self.video_transform = VideoTransform(mode=split, num_frames=self.num_frames)  # train or val model
        # for appear objects
        self.only_use_relevant_dets = True
        if self.only_use_relevant_dets:
            self.relevant_dets = []  # resort the detection numbers
            self.relevant_dets_classes = []

    def _load_metadata(self):
        # download specific
        metadata_dir = './meta_data/vcr1annots'
        split_files = {
            'train': 'train.jsonl',
            'val': 'val.jsonl',            # there is no test
            'test': 'test.jsonl'
        }
        target_split_fp = split_files[self.split]
        metadata = pd.read_json(os.path.join(metadata_dir, target_split_fp), lines=True)
        self.metadata = metadata

    def _get_image_path(self, sample):
        # print(sample.keys())
        # print(sample['img_fn'])
        # VCR/vcr1images
        rel_fp = sample['img_fn']
        return os.path.join(self.data_dir, 'vcr1images', rel_fp), rel_fp

    def get_objects(self, sample):
        metadata2 = pd.read_json(os.path.join(self.data_dir, 'vcr1images',
                                              sample['metadata_fn']), lines=True)
        object_meta = metadata2.iloc[0]
        return object_meta

    def _get_caption(self, sample):
        return sample[0]

    # def _get_objects(self, sample):
    #     metadata2 = pd.read_json(os.path.join(self.data_dir,
    #                                           sample['metadata_fn']), lines=True)
    #     sample = metadata2.iloc[0]
    #     return sample['boxes']

    def get_raw_image(self, sample, object_meta, img_color_mask=True):
        # print(sample)
        abs_fp, rel_fp = self._get_image_path(sample)
        # img = Image.open(abs_fp)
        img = cv2.imread(abs_fp)
        # add bbox annotation here
        if img_color_mask:
            img = color_img(img, object_meta, self.relevant_dets, self.only_use_relevant_dets)
        if img is None:
            raise Exception("Invalid img!", rel_fp)
        else:
            return img

    def get_image(self, index, sample, object_meta, image_key="image"):
        frames = []
        image = self.get_raw_image(sample, object_meta)
        frame = torch.from_numpy(image).byte()
        frame = frame.permute(2, 0, 1)
        frames.append(frame)
        frames = torch.stack(frames).permute(1, 0, 2, 3)
        # print(frames.size())
        image_tensor = [self.video_transform(frames).permute(1, 0, 2, 3)]  # to tchw
        # image = self.get_raw_image(sample)
        # image_tensor = [tr(image) for tr in self.transforms]
        # print(image_tensor.size())
        # image_tensor.unsqueeze(0)
        return image_tensor

    def get_false_image(self, rep, image_key="image"):
        random_index = random.randint(0, len(self.metadata) - 1)
        sample = self.metadata.iloc[random_index]
        image = self.get_raw_image(sample)
        image_tensor = [tr(image) for tr in self.transforms]
        return {f"false_image_{rep}": image_tensor}

    # def get_text(self, sample, object_meta):
    #     question = self.get_question(sample, object_meta)
    #     texts = []
    #     for answer in sample['answer_choices']:
    #         raw_text = question + '[SEP]'
    #         for word in answer:
    #             if isinstance(word, list):
    #                 for object_idx in word:
    #                     self.relevant_dets.add(object_idx)
    #                     rel_object_idx = object_idx
    #                     if self.only_use_relevant_dets:
    #                         rel_object_idx = len(self.relevant_dets) - 1  # begin from 0
    #                     raw_text += ' ' + object_meta['names'][object_idx] + ' ' + str(rel_object_idx)
    #             else:
    #                 raw_text += ' ' + word
    #         # for index in range(len(answer)):
    #         #     raw_text += ' ' + str(answer[index])
    #         print(raw_text)
    #         encoding = self.tokenizer(
    #             raw_text,
    #             padding="max_length",
    #             truncation=True,
    #             max_length=self.max_text_len,
    #             return_special_tokens_mask=True,
    #         )
    #         texts.append((raw_text, encoding))
    #     return texts

    def update_rele_det(self, sample, object_meta, index):
        text = []
        for i in range(len(sample['question'])):
            text.append(sample['question'][i])
        for i in range(len(sample['answer_choices'])):
            for j in range(len(sample['answer_choices'][i])):
                text.append(sample['answer_choices'][i][j])
        for i in range(len(sample['rationale_choices'])):
            for j in range(len(sample['rationale_choices'][i])):
                text.append(sample['rationale_choices'][i][j])
        # update relevant detes
        for word in text:
            if isinstance(word, list):
                for object_idx in word:
                    # self.relevant_dets.add(object_idx)
                    if object_idx not in self.relevant_dets:
                        self.relevant_dets.append(object_idx)
        for object in self.relevant_dets:
            self.relevant_dets_classes.append(object_meta['names'][object])
        # print(index, text)
        # print(index, self.relevant_dets)
        # print(index, self.relevant_dets_classes)
        #
        return text

    def get_text(self, sample, object_meta, index):
        # detect all object index and sort these items
        if self.only_use_relevant_dets:
            self.update_rele_det(sample, object_meta, index)
        question = self.get_question(sample, object_meta)
        qa_texts = []
        # image size: 384 x 384
        # prompt: [START] + "answer_question:"
        # prompt: [START] + ' provide rationale:'),
        # add all text tokens into this model.
        for answer in sample['answer_choices']:
            raw_text = question + 'answer question: '
            for word in answer:
                if isinstance(word, list):
                    for object_idx in word:
                        raw_text += ' ' + object_meta['names'][object_idx] + ' '
                        # rename the object index, for example
                        if self.only_use_relevant_dets:
                            raw_text += str(self.relevant_dets.index(object_idx))
                        else:
                            raw_text += str(object_idx)
                else:
                    raw_text += ' ' + word
            raw_text += '[END]'
            # print(index, raw_text)
            qa_encoding = self.tokenizer(
                raw_text,
                padding="max_length",
                truncation=True,
                max_length=self.max_text_len,
                return_special_tokens_mask=True,
            )
            qa_texts.append((raw_text, qa_encoding))

        gt_ans = sample['answer_choices'][sample['answer_label']]
        gt_ans_text = ""
        for word in gt_ans:
            if isinstance(word, list):
                for object_idx in word:
                    gt_ans_text += ' ' + object_meta['names'][object_idx] + ' '
                    # rename the object index, for example
                    if self.only_use_relevant_dets:
                        gt_ans_text += str(self.relevant_dets.index(object_idx))
                    else:
                        gt_ans_text += str(object_idx)
            else:
                gt_ans_text += ' ' + word
        qar_texts = []
        for reason in sample['rationale_choices']:
            raw_text = question + gt_ans_text + 'provide rationale: '
            for word in reason:
                if isinstance(word, list):
                    for object_idx in word:
                        raw_text += ' ' + object_meta['names'][object_idx] + ' '
                        if self.only_use_relevant_dets:
                            raw_text += str(self.relevant_dets.index(object_idx))
                        else:
                            raw_text += str(object_idx)
                else:
                    raw_text += ' ' + word
            # print(index, raw_text)
            raw_text += '[END]'
            encoding = self.tokenizer(
                raw_text,
                padding="max_length",
                truncation=True,
                max_length=self.max_text_len,
                return_special_tokens_mask=True,
            )
            qar_texts.append((raw_text, encoding))
        return [qa_texts, qar_texts]
    #
    # def get_qar(self, sample, object_meta):
    #     question = self.get_question(sample, object_meta) + '[SEP]' # '[MIDDLE]'
    #     gt_ans = sample['answer_choices'][sample['answer_label']]
    #     gt_ans_text = ""
    #     # for index in range(len(gt_ans)):
    #         # gt_ans_text += ' ' + str(gt_ans[index])
    #     for word in gt_ans:
    #         if isinstance(word, list):
    #             for object_idx in word:
    #                 self.relevant_dets.add(object_idx)
    #                 rel_object_idx = object_idx
    #                 if self.only_use_relevant_dets:
    #                     rel_object_idx = len(self.relevant_dets) - 1  # begin from 0
    #                 print(object_idx, rel_object_idx)
    #                 gt_ans_text += ' ' + object_meta['names'][object_idx] + ' ' + str(rel_object_idx)
    #         else:
    #             gt_ans_text += ' ' + word
    #     texts = []
    #     for reason in sample['rationale_choices']:
    #         raw_text = question + gt_ans_text + '[SEP]'
    #         for word in reason:
    #             if isinstance(word, list):
    #                 for object_idx in word:
    #                     self.relevant_dets.add(object_idx)
    #                     rel_object_idx = object_idx
    #                     if self.only_use_relevant_dets:
    #                         rel_object_idx = len(self.relevant_dets) - 1  # begin from 0
    #                     raw_text += ' ' + object_meta['names'][object_idx] + ' ' + str(rel_object_idx)
    #             else:
    #                 raw_text += ' ' + word
    #         print(raw_text)
    #         # for index in range(len(reason)):
    #         #     raw_text += ' ' + str(reason[index])
    #         #     self.relevant_dets.append(object_idx)
    #         # print(raw_text)
    #         encoding = self.tokenizer(
    #             raw_text,
    #             padding="max_length",
    #             truncation=True,
    #             max_length=self.max_text_len,
    #             return_special_tokens_mask=True,
    #         )
    #         texts.append((raw_text, encoding))
    #     return texts

    def get_answer_label(self, sample):
        answer = int(sample['answer_label'])
        return answer

    def get_reason_answer_label(self, sample):
        answer = int(sample['rationale_label'])
        return answer

    def get_question(self, sample, object_meta):
        raw_text = ""
        for index in range(len(sample['question'])):
            if isinstance(sample['question'][index], list):
                for object_idx in sample['question'][index]:
                    raw_text += ' ' + object_meta['names'][object_idx] + ' '
                    if self.only_use_relevant_dets:
                        raw_text += str(self.relevant_dets.index(object_idx))
                    else:
                        raw_text += str(object_idx)
            else:
                raw_text += ' ' + str(sample['question'][index])
        return raw_text

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, index):
        sample = self.metadata.iloc[index]
        object_meta = self.get_objects(sample)
        self.relevant_dets = []  # initalize
        self.relevant_dets_classes = []
        answer = self.get_answer_label(sample)
        reason_answer = self.get_reason_answer_label(sample)
        ret = {
            "img_index": index,
            "cap_index": index,
            "raw_index": index,
            'answer': answer,
            'reason_answer': reason_answer
        }
        # texts = self.get_text(sample, object_meta)
        # qar_texts = self.get_qar(sample, object_meta)
        [qa_texts, qar_texts] = self.get_text(sample, object_meta, index)
        ret["text"] = qa_texts[0]
        # print(texts[0])
        # update other answers as false text
        for i in range(self.draw_options_text - 1):
            ret.update({f"options_text_{i}": qa_texts[i+1]})
        for j in range(self.draw_options_text):
            ret.update({f"qar_text_{j}": qar_texts[j]})
        # print(ret.keys())
        image_tensor = self.get_image(index, sample, object_meta)
        ret["image"] = image_tensor
        return ret

