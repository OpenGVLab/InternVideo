from .video_base_dataset import BaseDataset, read_frames_from_img_dir
import random
import os
import pandas as pd


class ActivityNetDataset(BaseDataset):
    def __init__(self, *args, split="", **kwargs):
        assert split in ["train", "val", "test"]
        self.split = split
        self.metadata = None
        if split == "train":
            names = ["activitynet_train"]
        elif split == "val":
            names = ["activitynet_val"]
        elif split == "test":
            names = ["activitynet_val"]
        self._load_metadata()
        super().__init__(*args, **kwargs, names=names, text_column_name="caption")

    def _load_metadata(self):
        metadata_dir = './meta_data/activitynet'
        split_files = {
            'train': 'train.jsonl',
            'val': 'val1.jsonl',
            'test': 'val2.jsonl'
        }
        target_split_fp = split_files[self.split]
        metadata = pd.read_json(os.path.join(metadata_dir, target_split_fp), lines=True)
        self.metadata = metadata

    def _get_video_path(self, sample):
        rel_video_fp = sample['clip_name']
        full_video_fp = os.path.join(self.data_dir, 'activitynet_frames', rel_video_fp)
        return full_video_fp, rel_video_fp

    def get_raw_video(self, sample):
        abs_fp, rel_fp = self._get_video_path(sample)
        imgs, idxs, vlen = read_frames_from_img_dir(abs_fp, self.num_frames, mode=self.split)
        if imgs is None:
            raise Exception("Invalid img!", rel_fp)
        else:
            return imgs

    def get_video(self, index, sample, image_key="image"):
        videos = self.get_raw_video(sample)
        videos_tensor = self.video_aug(videos, self.video_transform)
        return {
            "video": videos_tensor,
            "vid_index": index,
            "cap_index": index,
            "raw_index": index,
        }

    def get_false_video(self, rep, image_key="image"):
        random_index = random.randint(0, len(self.metadata) - 1)
        sample = self.metadata.iloc[random_index]
        videos = self.get_raw_video(sample)
        videos_tensor = self.video_aug(videos, self.video_transform)
        return {f"false_video_{rep}": videos_tensor}

    def get_text(self, raw_index, sample):
        text = sample['caption']
        # print(text)
        encoding = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_text_len,
            return_special_tokens_mask=True,
        )
        # print(encoding.size())
        return {
            "text": (text, encoding),
            "img_index": raw_index,
            "cap_index": raw_index,
            "raw_index": raw_index,
        }

    def get_false_text(self, rep):
        random_index = random.randint(0, len(self.metadata) - 1)
        sample = self.metadata.iloc[random_index]
        text = sample['caption']
        encoding = self.tokenizer(
            text,
            # padding="max_length",
            truncation=True,
            max_length=self.max_text_len,
            return_special_tokens_mask=True,
        )
        return {f"false_text_{rep}": (text, encoding)}

    def get_suite(self, index):
        result = None
        while result is None:
            sample = self.metadata.iloc[index]
            try:
                ret = dict()
                ret.update(self.get_video(index, sample))
                if not self.image_only:
                    txt = self.get_text(index, sample)
                    ret.update({"replica": True if txt["cap_index"] > 0 else False})
                    ret.update(txt)

                for i in range(self.draw_false_image):
                    ret.update(self.get_false_video(i))
                for i in range(self.draw_false_text):
                    ret.update(self.get_false_text(i))
                result = True
            except Exception as e:
                print(f"Error while read file idx {sample.name} in {self.names[0]} -> {e}")
                index = random.randint(0, len(self.metadata) - 1)
        return ret

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, index):
        return self.get_suite(index)