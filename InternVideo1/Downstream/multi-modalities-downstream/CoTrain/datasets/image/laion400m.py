import json
from .base_dataset import BaseDataset
import random
import os
import pandas as pd
import io
from PIL import Image
from CoTrain.datasets import client


class LAION400MDataset(BaseDataset):
    def __init__(self, *args, split="", **kwargs):
        assert split in ["train", "val", "test"]
        self.split = split
        self.metadata = None
        self._load_metadata()
        if split == "train":
            names = ["laion400m_train"]
        elif split == "val":
            names = ["laion400m_val"]
        elif split == "test":
            names = ["laion400m_val"]
        print(names, ": ", len(self.metadata), "samples in total.")
        super().__init__(*args, **kwargs, names=names, text_column_name="caption")
        self.data_dir = ""

    def _load_metadata(self):
        file_path = "/mnt/lustre/share_data/liyizhuo.vendor/datasets/LAION-400M-partial-meta.json"
        if self.split == "train":
            self.metadata = [json.loads(x) for x in open(file_path).readlines()[:-10]]
        else:
            self.metadata = [json.loads(x) for x in open(file_path).readlines()[-10:]]
        
        self.metadata = [(x['caption'], x['filename']) for x in self.metadata]

    def _get_image_path(self, sample):
        # print(sample[1])
        # rel_fp = str(sample[1]).split('/')[-1]
        # print(os.path.join(self.data_dir, rel_fp))
        rel_fp = sample[1]
        return os.path.join(self.data_dir, rel_fp), rel_fp

    def _get_caption(self, sample):
        return sample[0]

    def get_raw_image(self, sample):
        # print(sample)
        abs_fp, rel_fp = self._get_image_path(sample)
        if "s3://" in abs_fp:
            img_bytes = client.get(abs_fp)
            assert img_bytes is not None, "Get image failed from {}".format(img_bytes)
            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        else:
            img = Image.open(abs_fp).convert("RGB")
        if img is None:
            raise Exception("Invalid img!", rel_fp)
        else:
            return img

    def _get_object_path(self, sample):
        """
        get the object npy path
        Args:
            sample (dict):
        Returns:
            abs path
        """
        rel_object_fp = os.path.join(sample[1], '1.npz')
        full_object_fp = os.path.join(self.object_dir, self.split, rel_object_fp)
        return os.path.join(self.split, rel_object_fp), full_object_fp

    def get_image(self, index, sample, image_key="image"):
        image = self.get_raw_image(sample)
        image_tensor = self.image_aug(image, self.transforms)
        # image_tensor = [tr(image).unsqueeze(0) for tr in self.transforms]
        return {
            "video": image_tensor,
            "vid_index": sample[1],
            "cap_index": index,
            "raw_index": index,
        }

    def get_false_image(self, rep, image_key="image"):
        random_index = random.randint(0, len(self.metadata) - 1)
        sample = self.metadata[random_index]
        image = self.get_raw_image(sample)
        #image_tensor = [tr(image).unsqueeze(0) for tr in self.transforms]
        image_tensor = self.image_aug(image, self.transforms)
        return {f"false_video_{rep}": image_tensor}

    def get_text(self, raw_index, sample):
        text = sample[0]
        encoding = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_text_len,
            return_special_tokens_mask=True,
        )
        return {
            "text": (text, encoding),
            "vid_index": sample[1],
            "cap_index": raw_index,
            "raw_index": raw_index,
        }

    def get_false_text(self, rep):
        random_index = random.randint(0, len(self.metadata) - 1)
        sample = self.metadata[random_index]
        text = sample[0]
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_text_len,
            return_special_tokens_mask=True,
        )
        return {f"false_text_{rep}": (text, encoding)}

    def get_suite(self, index):
        result = None
        # print(self.draw_false_image) # 1
        while result is None:
            sample = self.metadata[index]
            # print(sample)
            try:
                ret = dict()
                ret.update(self.get_image(index, sample))
                if not self.image_only:
                    txt = self.get_text(index, sample)
                    ret.update({"replica": True if txt["cap_index"] > 0 else False})
                    ret.update(txt)

                for i in range(self.draw_false_image):
                    ret.update(self.get_false_image(i))
                for i in range(self.draw_false_text):
                    ret.update(self.get_false_text(i))
                result = True
            except Exception as e:
                print(f"Error while read file idx {sample[1]} in {self.names[0]} -> {e}")
                index = random.randint(0, len(self.metadata) - 1)
            # ret["image"] = ret["image"].unsqueeze(1)
        return ret

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, index):
        return self.get_suite(index)

