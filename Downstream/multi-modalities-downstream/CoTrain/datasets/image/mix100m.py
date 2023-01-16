import json
from .base_dataset import BaseDataset
import random
import os
import pandas as pd
import numpy as np
import io
import torch
from PIL import Image
from CoTrain.datasets import client
import CoTrain.modules.dist_utils as du


class MIX100MDataset(BaseDataset):
    def __init__(self, *args, split="", **kwargs):
        assert split in ["train", "val", "test"]
        self.split = split
        self.metadata = None
        self.world_size = du.get_world_size()
        self.rank = du.get_rank()
        self._load_metadata()
        if split == "train":
            names = ["mix100m_train"]
        elif split == "val":
            names = ["mix100m_val"]
        elif split == "test":
            names = ["mix100m_val"]
        print(names, ": ", len(self.metadata), "samples in total.")
        super().__init__(*args, **kwargs, names=names, text_column_name="caption")
        self.data_dir = ""

    def _load_metadata(self):
        if self.split != "train":
            file_path = "/mnt/lustre/share_data/liyizhuo/datasets/fake_mix100m_val.json"
            self.metadata = [json.loads(x) for x in open(file_path).readlines()]
            self.metadata = [
                (" ".join(x["caption"]), os.path.join(x["image_root"], x["filename"]))
                for x in self.metadata
            ]
            return None

        meta_root = (
            "s3://liyizhuo/datasets/shlab_softmax_100m_10000/"
        )
        file_list = [os.path.join(meta_root, f"{i}".zfill(5) + ".json") for i in range(10)]

        ranked_meta = [[] for _ in range(self.world_size)]
        ranked_num = [0 for _ in range(self.world_size)]
        import time

        start_time = time.time()
        count = 0
        for seed_num, each_meta_file in enumerate(file_list):
            f = client.get(each_meta_file).decode().strip().split("\n")
            np.random.seed(seed_num)
            random_ranks = np.random.randint(0, self.world_size, size=(len(f), ))
            for i, line in enumerate(f):
                count += 1
                if self.rank == random_ranks[i]:
                    info = json.loads(line.encode("UTF-8"))
                    info = (
                        " ".join(info["caption"]),
                        os.path.join(info["image_root"], info["filename"]),
                    )

                    ranked_meta[self.rank].append(info)
                    ranked_num[self.rank] += 1
                if count % 1000000 == 0 and self.rank == 0:
                    print(
                        "-------------------------------------------------------------- every 1M time:",
                        (time.time() - start_time),
                        "{}M".format(count / 1000000),
                    )
            del f

        self.metadata = ranked_meta[self.rank]
        num = ranked_num[self.rank]

        # balance data length in each subprocess
        ranked_num = du.all_gather(num)
        du.synchronize()
        max_num = max(ranked_num)
        if max_num > num:
            diff = max_num - num
            self.metadata.extend(random.sample(self.metadata, diff))
            num = len(self.metadata)
        assert num == max_num

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
        rel_object_fp = os.path.join(sample[1], "1.npz")
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
        # image_tensor = [tr(image).unsqueeze(0) for tr in self.transforms]
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
        index %= len(self.metadata)
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
                print(
                    f"Error while read file idx {sample[1]} in {self.names[0]} -> {e}"
                )
                index = random.randint(0, len(self.metadata) - 1)
            # ret["image"] = ret["image"].unsqueeze(1)
        return ret

    def __len__(self):
        return len(self.metadata) * self.world_size

    def __getitem__(self, index):
        return self.get_suite(index)
