import random
import torch
import io
import pyarrow as pa
import os
import cv2
import numpy as np
from PIL import Image
from CoTrain.transforms import keys_to_transforms
import decord
from CoTrain.transforms.image.imageaug import image_aug
import CoTrain.modules.InternVideo as internvideo


class BaseDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_dir: str,
        transform_keys: list,
        image_size: int,
        names: list,
        text_column_name: str = "",
        remove_duplicate=True,
        max_text_len=40,
        draw_false_image=0,
        draw_false_video=0,
        draw_false_text=0,
        image_only=False,
        num_frames=1,
        draw_options_text=0,
        backend='v100'
    ):
        """
        data_dir : where dataset file *.arrow lives; existence should be guaranteed via DataModule.prepare_data
        transform_keys : keys for generating augmented views of images
        text_column_name : pyarrow table column name that has list of strings as elements
        """
        assert len(transform_keys) >= 1
        super().__init__()

        self.transforms = keys_to_transforms(transform_keys, size=image_size, mode=self.split)
        self.image_aug = image_aug
        self.text_column_name = text_column_name
        self.names = names
        self.max_text_len = max_text_len
        self.draw_false_image = draw_false_image
        self.draw_false_text = draw_false_text
        self.image_only = image_only
        self.data_dir = data_dir
        self.draw_options_text = draw_options_text
        if torch.distributed.get_rank() == 0:
            print('*'*100)
            print("image sub datasets: {}".format(names))
        # print(names)
        split_name = None
        if len(names) != 0:
            self.data_dir = os.path.join(self.data_dir, names[0].split('_')[0])  # e.g. coco_train -> coco
            split_name = names[0].split('_')[0]
        if torch.distributed.get_rank() == 0:
            print(self.data_dir)
        if split_name and split_name in ['msrvtt', 'cc3m', 'vcr', 'cc12m', 'yfcc15m', 'laion400m', 'mix100m']:
            if torch.distributed.get_rank() == 0:
                print("no arrow available for {}, load from disk".format(names[0]))
        else:
            if len(names) != 0:
                tables = [
                    pa.ipc.RecordBatchFileReader(
                        pa.memory_map(f"{self.data_dir}/{name}.arrow", "r")
                    ).read_all()
                    for name in names
                    if os.path.isfile(f"{self.data_dir}/{name}.arrow")
                ]
                # print(names, tables)
                self.table_names = list()
                for i, name in enumerate(names):
                    self.table_names += [name] * len(tables[i])

                self.table = pa.concat_tables(tables, promote=True)
                if text_column_name != "":
                    self.text_column_name = text_column_name
                    self.all_texts = self.table[text_column_name].to_pandas().tolist()
                    self.all_texts = (
                        [list(set(texts)) for texts in self.all_texts]
                        if remove_duplicate
                        else self.all_texts
                    )
                else:
                    self.all_texts = list()
            else:
                self.all_texts = list()

            self.index_mapper = dict()

            if text_column_name != "" and not self.image_only:
                j = 0
                for i, texts in enumerate(self.all_texts):
                    for _j in range(len(texts)):
                        self.index_mapper[j] = (i, _j)
                        j += 1
            else:
                for i in range(len(self.table)):
                    self.index_mapper[i] = (i, None)

        #
        # if len(names) != 0:
        #     tables = [
        #         pa.ipc.RecordBatchFileReader(
        #             pa.memory_map(f"{data_dir}/{name}.arrow", "r")
        #         ).read_all()
        #         for name in names
        #         if os.path.isfile(f"{data_dir}/{name}.arrow")
        #     ]
        #
        #     self.table_names = list()
        #     for i, name in enumerate(names):
        #         self.table_names += [name] * len(tables[i])
        #
        #     self.table = pa.concat_tables(tables, promote=True)
        #     if text_column_name != "":
        #         self.text_column_name = text_column_name
        #         self.all_texts = self.table[text_column_name].to_pandas().tolist()
        #         self.all_texts = (
        #             [list(set(texts)) for texts in self.all_texts]
        #             if remove_duplicate
        #             else self.all_texts
        #         )
        #     else:
        #         self.all_texts = list()
        # else:
        #     self.all_texts = list()
        #
        # self.index_mapper = dict()
        #
        # if text_column_name != "" and not self.image_only:
        #     j = 0
        #     for i, texts in enumerate(self.all_texts):
        #         for _j in range(len(texts)):
        #             self.index_mapper[j] = (i, _j)
        #             j += 1
        # else:
        #     for i in range(len(self.table)):
        #         self.index_mapper[i] = (i, None)

    @property
    def corpus(self):
        return [text for texts in self.all_texts for text in texts]

    def __len__(self):
        return len(self.index_mapper)

    def get_raw_image(self, index, image_key="image"):
        index, caption_index = self.index_mapper[index]
        image_bytes = io.BytesIO(self.table[image_key][index].as_py())
        image_bytes.seek(0)
        return Image.open(image_bytes).convert("RGB")

    def get_image(self, index, image_key="image"):
        image = self.get_raw_image(index, image_key=image_key)
        # image_tensor = [tr(image).unsqueeze(0) for tr in self.transforms]
        image_tensor = self.image_aug(image, self.transforms)
        return {
            "video": image_tensor,
            "vid_index": self.index_mapper[index][0],
            "cap_index": self.index_mapper[index][1],
            "raw_index": index,
        }

    def get_false_image(self, rep, image_key="image"):
        random_index = random.randint(0, len(self.index_mapper) - 1)
        image = self.get_raw_image(random_index, image_key=image_key)
        # image_tensor = [tr(image).unsqueeze(0) for tr in self.transforms]
        image_tensor = self.image_aug(image, self.transforms)
        return {f"false_video_{rep}": image_tensor}

    def get_text(self, raw_index):
        index, caption_index = self.index_mapper[raw_index]

        text = self.all_texts[index][caption_index]
        encoding = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_text_len,
            return_special_tokens_mask=True,
        )
        return {
            "text": (text, encoding),
            "vid_index": index,
            "cap_index": caption_index,
            "raw_index": raw_index,
        }

    def get_false_text(self, rep):
        random_index = random.randint(0, len(self.index_mapper) - 1)
        index, caption_index = self.index_mapper[random_index]
        text = self.all_texts[index][caption_index]
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_text_len,
            return_special_tokens_mask=True,
        )
        return {f"false_text_{rep}": (text, encoding)}

    def get_suite(self, index):
        result = None
        while result is None:
            try:
                ret = dict()
                ret.update(self.get_image(index))
                if not self.image_only:
                    txt = self.get_text(index)
                    ret.update({"replica": True if txt["cap_index"] > 0 else False})
                    ret.update(txt)

                for i in range(self.draw_false_image):
                    ret.update(self.get_false_image(i))
                for i in range(self.draw_false_text):
                    ret.update(self.get_false_text(i))
                result = True
            except Exception as e:
                print(f"Error while read file idx {index} in {self.names[0]} -> {e}")
                index = random.randint(0, len(self.index_mapper) - 1)
        return ret

    def collate(self, batch, mlm_collator):
        batch_size = len(batch)
        keys = set([key for b in batch for key in b.keys()])
        dict_batch = {k: [dic[k] if k in dic else None for dic in batch] for k in keys}
        # print(dict_batch)

        img_keys = [k for k in list(dict_batch.keys()) if "video" in k]
        img_sizes = list()

        for img_key in img_keys:
            img_sizes += [ii.shape for i in dict_batch[img_key] if i is not None for ii in i]

        for size in img_sizes:
            assert (
                len(size) == 4
            ), f"Collate error, an image should be in shape of (N, 3,  H, W), instead of given {size}"

        if len(img_keys) != 0:
            global_max_height = max([i[2] for i in img_sizes])
            global_max_width = max([i[3] for i in img_sizes])
        for img_key in img_keys:
            img = dict_batch[img_key]
            view_size = len(dict_batch[img_key][0])
            new_images = [
                    torch.zeros(batch_size, 1, 3, global_max_height, global_max_width)
                    for _ in range(view_size)
                ]
            # print(len(img))
            for bi in range(batch_size):
                orig_batch = img[bi]
                for vi in range(view_size):
                    if orig_batch is None:
                        continue
                    else:
                        orig = img[bi][vi]
                        new_images[vi][bi, :, :, : orig.shape[2], : orig.shape[3]] = orig

            dict_batch[img_key] = new_images

        txt_keys = [k for k in list(dict_batch.keys()) if "text" in k]

        if len(txt_keys) != 0:
            texts = [[d[0] for d in dict_batch[txt_key]] for txt_key in txt_keys]
            encodings = [[d[1] for d in dict_batch[txt_key]] for txt_key in txt_keys]
            draw_text_len = len(encodings)
            flatten_encodings = [e for encoding in encodings for e in encoding]
            flatten_mlms = mlm_collator(flatten_encodings)

            for i, txt_key in enumerate(txt_keys):
                texts, encodings = (
                    [d[0] for d in dict_batch[txt_key]],
                    [d[1] for d in dict_batch[txt_key]],
                )

                mlm_ids, mlm_labels = (
                    flatten_mlms["input_ids"][batch_size * (i) : batch_size * (i + 1)],
                    flatten_mlms["labels"][batch_size * (i) : batch_size * (i + 1)],
                )

                input_ids = torch.zeros_like(mlm_ids)
                attention_mask = torch.zeros_like(mlm_ids)
                for _i, encoding in enumerate(encodings):
                    _input_ids, _attention_mask = (
                        torch.tensor(encoding["input_ids"]),
                        torch.tensor(encoding["attention_mask"]),
                    )
                    input_ids[_i, : len(_input_ids)] = _input_ids
                    attention_mask[_i, : len(_attention_mask)] = _attention_mask

                dict_batch[txt_key] = texts
                dict_batch[f"{txt_key}_ids"] = input_ids
                dict_batch[f"{txt_key}_labels"] = torch.full_like(input_ids, -100)
                dict_batch[f"{txt_key}_ids_mlm"] = mlm_ids
                dict_batch[f"{txt_key}_labels_mlm"] = mlm_labels
                dict_batch[f"{txt_key}_masks"] = attention_mask

            clip_text_ids, clip_special_tokens_mask = internvideo.tokenize(
                dict_batch["text"], truncate=True, return_special_tokens_mask=True)
            dict_batch["clip_text_ids"] = clip_text_ids
            dict_batch["clip_special_tokens_mask"] = clip_special_tokens_mask
        
        return dict_batch


    # def collate(self, batch, mlm_collator):
    #     batch_size = len(batch)
    #     keys = set([key for b in batch for key in b.keys()])
    #     dict_batch = {k: [dic[k] if k in dic else None for dic in batch] for k in keys}
    #     # print(dict_batch)
    #
    #     img_keys = [k for k in list(dict_batch.keys()) if "video" in k]
    #     img_sizes = list()
    #
    #     for img_key in img_keys:
    #         img_sizes += [ii.shape for i in dict_batch[img_key][0] if i is not None for ii in i]
    #
    #     for size in img_sizes:
    #         assert (
    #             len(size) == 4
    #         ), f"Collate error, an image should be in shape of (N, 3,  H, W), instead of given {size}"
    #
    #     if len(img_keys) != 0:
    #         global_max_height = max([i[2] for i in img_sizes])
    #         global_max_width = max([i[3] for i in img_sizes])
    #         local_max_height = min([i[2] for i in img_sizes])
    #         local_max_width = min([i[3] for i in img_sizes])
    #     for img_key in img_keys:
    #         img = dict_batch[img_key]
    #         global_view_size = len(dict_batch[img_key][0][0])
    #         local_view_size = len(dict_batch[img_key][0][1])
    #         # for image, padding one time dimension
    #         new_images = [
    #             [
    #                 torch.zeros(batch_size, 1, 3, global_max_height, global_max_width)
    #                 for _ in range(global_view_size)
    #             ],
    #             [
    #                 torch.zeros(batch_size, 1, 3, local_max_height, local_max_width)
    #                 for _ in range(local_view_size)
    #             ]
    #         ]
    #         # print(len(img))
    #         for bi in range(batch_size):
    #             orig_batch = img[bi]
    #             for vi in range(global_view_size):
    #                 if orig_batch is None:
    #                     continue
    #                 else:
    #                     orig = img[bi][0][vi]
    #                     new_images[0][vi][bi, :, :, : orig.shape[2], : orig.shape[3]] = orig
    #
    #         for bi in range(batch_size):
    #             orig_batch = img[bi]
    #             for vi in range(local_view_size):
    #                 if orig_batch is None:
    #                     continue
    #                 else:
    #                     orig = img[bi][1][vi]
    #                     new_images[1][vi][bi, :, :, : orig.shape[2], : orig.shape[3]] = orig
    #
    #         dict_batch[img_key] = new_images
    #
    #     txt_keys = [k for k in list(dict_batch.keys()) if "text" in k]
    #
    #     if len(txt_keys) != 0:
    #         texts = [[d[0] for d in dict_batch[txt_key]] for txt_key in txt_keys]
    #         encodings = [[d[1] for d in dict_batch[txt_key]] for txt_key in txt_keys]
    #         draw_text_len = len(encodings)
    #         flatten_encodings = [e for encoding in encodings for e in encoding]
    #         flatten_mlms = mlm_collator(flatten_encodings)
    #
    #         for i, txt_key in enumerate(txt_keys):
    #             texts, encodings = (
    #                 [d[0] for d in dict_batch[txt_key]],
    #                 [d[1] for d in dict_batch[txt_key]],
    #             )
    #
    #             mlm_ids, mlm_labels = (
    #                 flatten_mlms["input_ids"][batch_size * (i) : batch_size * (i + 1)],
    #                 flatten_mlms["labels"][batch_size * (i) : batch_size * (i + 1)],
    #             )
    #
    #             input_ids = torch.zeros_like(mlm_ids)
    #             attention_mask = torch.zeros_like(mlm_ids)
    #             for _i, encoding in enumerate(encodings):
    #                 _input_ids, _attention_mask = (
    #                     torch.tensor(encoding["input_ids"]),
    #                     torch.tensor(encoding["attention_mask"]),
    #                 )
    #                 input_ids[_i, : len(_input_ids)] = _input_ids
    #                 attention_mask[_i, : len(_attention_mask)] = _attention_mask
    #
    #             dict_batch[txt_key] = texts
    #             dict_batch[f"{txt_key}_ids"] = input_ids
    #             dict_batch[f"{txt_key}_labels"] = torch.full_like(input_ids, -100)
    #             dict_batch[f"{txt_key}_ids_mlm"] = mlm_ids
    #             dict_batch[f"{txt_key}_labels_mlm"] = mlm_labels
    #             dict_batch[f"{txt_key}_masks"] = attention_mask
    #
    #     return dict_batch


def sample_frames(num_frames, vlen, sample='rand', fix_start=None):
    acc_samples = min(num_frames, vlen)
    intervals = np.linspace(start=0, stop=vlen, num=acc_samples + 1).astype(int)
    ranges = []
    for idx, interv in enumerate(intervals[:-1]):
        ranges.append((interv, intervals[idx + 1] - 1))
    if sample == 'rand':
        frame_idxs = [random.choice(range(x[0], x[1])) for x in ranges]
    elif fix_start is not None:
        frame_idxs = [x[0] + fix_start for x in ranges]
    elif sample == 'uniform':
        frame_idxs = [(x[0] + x[1]) // 2 for x in ranges]
    else:
        raise NotImplementedError

    return frame_idxs


def read_frames_cv2(video_path, num_frames, sample='rand', fix_start=None):
    cap = cv2.VideoCapture(video_path)
    assert (cap.isOpened())
    vlen = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # get indexes of sampled frames
    frame_idxs = sample_frames(num_frames, vlen, sample=sample, fix_start=fix_start)
    frames = []
    success_idxs = []
    for index in frame_idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, index - 1)
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = torch.from_numpy(frame).byte()
            # # (H x W x C) to (C x H x W)
            frame = frame.permute(2, 0, 1)
            frames.append(frame)
            success_idxs.append(index)
        else:
            pass
            # print(frame_idxs, ' fail ', index, f'  (vlen {vlen})')
    # return frames tensor
    frames = torch.stack(frames) # .float() / 255
    # print(frames.size())
    cap.release()
    return frames, success_idxs, vlen


def read_frames_decord(video_path, num_frames, sample='rand', fix_start=None):
    video_reader = decord.VideoReader(video_path, num_threads=1)
    vlen = len(video_reader)
    frame_idxs = sample_frames(num_frames, vlen, sample=sample, fix_start=fix_start)
    frames = video_reader.get_batch(frame_idxs)
    frames = frames.float() / 255
    frames = frames.permute(0, 3, 1, 2)
    return frames, frame_idxs, vlen