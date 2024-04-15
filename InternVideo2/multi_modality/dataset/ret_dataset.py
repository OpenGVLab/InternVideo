from os.path import basename
import numpy as np
import logging
import torch

from dataset.base_dataset import BaseDataset
from dataset.utils import load_anno, pre_text
from dataset.video_utils import VIDEO_READER_FUNCS
from dataset.text_prompt import kinetics_templates_action_clip as kinetics_templates

logger = logging.getLogger(__name__)


class AudioTxtRetTrainDataset(BaseDataset):
    media_type = "audio"

    def __init__(
            self, ann_file, transform, audio_sample_rate, 
            audio_reader_type='librosa', max_audio_length=0, num_tries=3):
        super(AudioTxtRetTrainDataset, self).__init__()
        self.anno_list = load_anno(ann_file)
        self.transform = transform
        self.audio_reader_type = audio_reader_type
        self.num_tries = num_tries
        self.has_multi_audio_gt = ann_file.get("has_multi_audio_gt", False)
        self.trimmed30 = ann_file.get("trimmed30", False)
        
        self.max_audio_length = max_audio_length
        self.audio_sample_rate = audio_sample_rate
        self.match_ids = {}

        n = 0
        for ann in self.anno_list:
            key = ann["caption"] if self.has_multi_audio_gt else basename(ann["image"])
            if key not in self.match_ids:
                self.match_ids[key] = n
                n += 1

    def __len__(self):
        return len(self.anno_list)

    def __getitem__(self, index):
        try:
            ann = self.anno_list[index]
            audio, index = self.load_and_transform_media_data(index, ann['image'])
            caption = pre_text(ann["caption"])
            key = ann["caption"] if self.has_multi_audio_gt else basename(ann["image"])
            return audio, caption, self.match_ids[key]
        except Exception as e:
            logger.error(e)
            print(e, flush=True)
            index = np.random.randint(0, len(self))
            return self.__getitem__(index)


class AudioTxtRetEvalDataset(BaseDataset):
    media_type = "audio"

    def __init__(
            self, ann_file, transform, audio_sample_rate,
            audio_reader_type='librosa', max_audio_length=0, num_tries=3):
        super(AudioTxtRetEvalDataset, self).__init__()
        self.anno_list = load_anno(ann_file)
        self.transform = transform
        self.audio_sample_rate = audio_sample_rate
        self.max_audio_length = max_audio_length
        self.audio_reader_type = audio_reader_type
        self.num_tries = num_tries
        self.has_multi_audio_gt = ann_file.get("has_multi_audio_gt", False)
        self.trimmed30 = ann_file.get("trimmed30", False)
        self.max_txt_l = ann_file.get("max_txt_l", 32)

        self.text = None
        self.audio = None
        self.txt2img = None
        self.img2txt = None

        self.build_data()

    def build_data(self):
        self.text = []
        self.audio = []
        self.txt2img = {}
        self.img2txt = {}
        if self.has_multi_audio_gt:
            self.build_data_multi_audio_gt()
        else:
            self.build_data_multi_txt_gt()

    def build_data_multi_audio_gt(self):
        """each text may have multiple ground_truth audio, e.g., ssv2"""
        audio_id = 0
        for txt_id, ann in enumerate(self.anno_list):
            self.text.append(pre_text(ann["caption"]))
            self.txt2img[txt_id] = []
            _audios = ann["image"] \
                if isinstance(ann["image"], list) else [ann["image"], ]
            for i, audio in enumerate(_audios):
                self.audio.append(audio)
                self.txt2img[txt_id].append(audio_id)
                self.img2txt[audio_id] = txt_id
                audio_id += 1

    def build_data_multi_txt_gt(self):
        """each audio may have multiple ground_truth text, e.g., COCO and Flickr30K"""
        txt_id = 0
        for audio_id, ann in enumerate(self.anno_list):
            self.audio.append(ann["image"])
            self.img2txt[audio_id] = []
            _captions = ann["caption"] \
                if isinstance(ann["caption"], list) else [ann["caption"], ]
            for i, caption in enumerate(_captions):
                self.text.append(pre_text(caption))
                self.img2txt[audio_id].append(txt_id)
                self.txt2img[txt_id] = audio_id
                txt_id += 1

    def __len__(self):
        return len(self.anno_list)

    def __getitem__(self, index):
        ann = self.anno_list[index]
        audio, index = self.load_and_transform_media_data(index, ann["image"])
        return audio, index



class ImgTxtRetTrainDataset(BaseDataset):
    media_type = "image"

    def __init__(self, ann_file, transform):
        super(ImgTxtRetTrainDataset, self).__init__()
        self.anno_list = load_anno(ann_file)
        self.transform = transform
        # each caption has multiple image as ground_truth, e.g., ssv2
        self.has_multi_txt_gt = ann_file.get("has_multi_txt_gt", False)
        self.has_multi_vision_gt = ann_file.get("has_multi_vision_gt", False)

        if self.has_multi_txt_gt:
            logger.info("The dataset has multiple ground truth for a image/video!")
            tmp_anno_list = []
            for ann in self.anno_list:
                img_path = ann["image"]
                for caption in ann["caption"]:
                    tmp_anno_list.append({
                        "image": img_path,
                        "caption": caption
                    })
            self.anno_list = tmp_anno_list
            
        self.match_ids = {}
        n = 0
        for ann in self.anno_list:
            key = ann["caption"] if self.has_multi_vision_gt else basename(ann["image"])
            if key not in self.match_ids:
                self.match_ids[key] = n
                n += 1

    def __len__(self):
        return len(self.anno_list)

    def __getitem__(self, index):
        try:
            ann = self.anno_list[index]
            image, index = self.load_and_transform_media_data(index, ann["image"])
            caption = pre_text(ann["caption"])
            key = ann["caption"] if self.has_multi_vision_gt else basename(ann["image"])
            return image, caption, self.match_ids[key]
        except Exception as e:
            logger.error(e)
            print(e, flush=True)
            index = np.random.randint(0, len(self))
            return self.__getitem__(index)


class VidTxtRetTrainDataset(ImgTxtRetTrainDataset):
    media_type = "video"

    def __init__(
            self, ann_file, transform, num_frames=4,
            video_reader_type="decord", sample_type="rand", num_tries=3):
        super(VidTxtRetTrainDataset, self).__init__(ann_file, transform)
        self.num_frames = num_frames
        self.video_reader_type = video_reader_type
        self.video_reader = VIDEO_READER_FUNCS[video_reader_type]
        self.sample_type = sample_type
        self.num_tries = num_tries
        self.read_clip_from_video = ann_file.get("read_clip_from_video", False)
        if self.read_clip_from_video:
            raise NotImplementedError("key for match_ids is not implemented!")
        self.is_paragraph_retrieval = ann_file.get("is_paragraph_retrieval", False)
        if self.is_paragraph_retrieval:
            self.anno_list = preprocess_para_retrieval_data(self.anno_list)
        self.trimmed30 = ann_file.get("trimmed30", False)
        if self.trimmed30:
            logger.info("Trimming the video, only use the first 30s!")


class AudioVidTxtRetTrainDataset(VidTxtRetTrainDataset):
    media_type = "audio_video"

    def __init__(
            self, ann_file, transform, 
            audio_sample_rate=16000, 
            audio_reader_type='torchaudio',
            max_audio_length=10,
            num_frames=4,
            video_reader_type="decord", sample_type="rand", num_tries=3):
        super(AudioVidTxtRetTrainDataset, self).__init__(ann_file, transform,
            num_frames=num_frames, video_reader_type=video_reader_type, sample_type=sample_type, num_tries=num_tries)

        assert self.media_type == 'audio_video', self.media_type
        self.audio_sample_rate = audio_sample_rate
        self.audio_reader_type = audio_reader_type
        self.max_audio_length = max_audio_length

        self.has_multi_audio_gt = ann_file.get("has_multi_audio_gt", False)
        self.read_audio_from_video = ann_file.get("read_audio_from_video", False)
        self.zero_audio_padding_for_video = ann_file.get("zero_audio_padding_for_video", False)

    def __getitem__(self, index):
        try:
            ann = self.anno_list[index]
            caption = pre_text(ann["caption"])

            data_path = {'video': ann["image"]}
            data_path["read_clip_from_video"] = self.read_clip_from_video
            if "audio" in ann.keys():
                data_path["read_audio_from_video"] = False
                data_path["audio"] = ann["audio"]
            else:
                data_path["read_audio_from_video"] = self.read_audio_from_video
            
            media, index = self.load_and_transform_media_data(index, data_path)

            audio = media[0]
            if audio is None and self.zero_audio_padding_for_video:
                logger.warning(f"No audio in {data_path}")
                media[0] = torch.zeros((998, 64), dtype=torch.float32)

            key = ann["caption"] if self.has_multi_vision_gt else basename(ann["image"])
            return media, caption, self.match_ids[key]
        
        except Exception as e:
            logger.error(e)
            print(e, flush=True)
            index = np.random.randint(0, len(self))
            return self.__getitem__(index)
        

class ImgTxtRetEvalDataset(BaseDataset):
    media_type = "image"

    def __init__(self, ann_file, transform):
        super(ImgTxtRetEvalDataset, self).__init__()
        self.raw_anno_list = load_anno(ann_file)

        self.transform = transform
        self.has_multi_vision_gt = ann_file.get("has_multi_vision_gt", False)  # each caption has multiple image as ground_truth
        
        self.is_act_rec = ann_file.get("is_act_rec", False)
        self.max_txt_l = ann_file.get("max_txt_l", 32) # NOTE

        self.text = None
        self.image = None
        self.txt2img = None
        self.img2txt = None
        self.build_data()

    def build_data(self):
        self.text = []
        self.image = []
        self.txt2img = {}
        self.img2txt = {}
        if self.is_act_rec:
            self.build_data_act_rec()
        elif self.has_multi_vision_gt:
            self.build_data_multi_img_gt()
        else:
            self.build_data_multi_txt_gt()
        self.anno_list = [dict(image=e) for e in self.image]
    
    def build_data_act_rec(self):
        """action recognition task, e.g., kinetics400"""
        text = list(set([e["caption"] for e in self.raw_anno_list]))
        text2label = {e: i for i, e in enumerate(text)}
        text = [[t.format(e) for t in kinetics_templates] for e in text]
        text = [e for l in text for e in l]
        self.text = [pre_text(e) for e in text]
        self.num_prompts = len(kinetics_templates)
        self.img2txt = {i: text2label[e["caption"]] for i, e in enumerate(self.raw_anno_list)}
        self.txt2img = [[] for _ in range(len(text) // len(kinetics_templates))]
        for i, e in enumerate(self.raw_anno_list):
            self.image.append(e["image"])
            self.txt2img[text2label[e["caption"]]].append(i)
        logger.info(f"Action recognition, number of prompts: {self.num_prompts}")
        logger.info(f"Action recognition, number of classes: {len(self.text)}")

    def build_data_multi_img_gt(self):
        """each text may have multiple ground_truth image, e.g., ssv2"""
        img_id = 0
        for txt_id, ann in enumerate(self.raw_anno_list):
            self.text.append(pre_text(ann["caption"]))
            self.txt2img[txt_id] = []
            _images = ann["image"] \
                if isinstance(ann["image"], list) else [ann["image"], ]
            for i, image in enumerate(_images):
                self.image.append(image)
                self.txt2img[txt_id].append(img_id)
                self.img2txt[img_id] = txt_id
                img_id += 1

    def build_data_multi_txt_gt(self):
        """each image may have multiple ground_truth text, e.g., COCO and Flickr30K"""
        txt_id = 0
        for img_id, ann in enumerate(self.raw_anno_list):
            self.image.append(ann["image"])
            self.img2txt[img_id] = []
            _captions = ann["caption"] \
                if isinstance(ann["caption"], list) else [ann["caption"], ]
            for i, caption in enumerate(_captions):
                self.text.append(pre_text(caption))
                self.img2txt[img_id].append(txt_id)
                self.txt2img[txt_id] = img_id
                txt_id += 1

    def __len__(self):
        return len(self.anno_list)

    def __getitem__(self, index):
        ann = self.anno_list[index]
        image, index = self.load_and_transform_media_data(index, ann["image"])
        return image, index


class VidTxtRetEvalDataset(ImgTxtRetEvalDataset):
    media_type = "video"

    def __init__(
            self, ann_file, transform, num_frames=4,
            video_reader_type="decord", sample_type="rand", num_tries=1):
        super(VidTxtRetEvalDataset, self).__init__(ann_file, transform)
        self.num_frames = num_frames
        self.video_reader_type = video_reader_type
        self.video_reader = VIDEO_READER_FUNCS[video_reader_type]
        self.sample_type = sample_type
        self.num_tries = num_tries
        self.is_paragraph_retrieval = ann_file.get("is_paragraph_retrieval", False)
        if self.is_paragraph_retrieval:
            logger.info("Preprocess paragraph retrieval data!!!")
            self.anno_list = preprocess_para_retrieval_data(self.raw_anno_list)
        self.trimmed30 = ann_file.get("trimmed30", False)
        if self.trimmed30:
            logger.info("Trimming the video, only use the first 30s!!!")
        self.read_clip_from_video = ann_file.get("read_clip_from_video", False)
        self.use_subtitle = ann_file.get("use_subtitle", False)
        if self.use_subtitle:
            if self.is_act_rec:
                raise NotImplementedError
            self.build_subtitle_data()

        self.build_data()

    def __getitem__(self, index):
        ann = self.anno_list[index]
        if self.read_clip_from_video:
            raise NotImplementedError("key for match_ids is not implemented!")
        else:
            data_path = ann["image"]
        image, index = self.load_and_transform_media_data(index, data_path)
        return image, index

    def build_subtitle_data(self):
        self.subtitle = []
        for _, ann in enumerate(self.raw_anno_list):
            if self.trimmed30:
                if "asr_trimmed_30" in ann.keys():
                    self.subtitle.append(pre_text(ann["asr_trimmed_30"]))
                else:
                    self.subtitle.append("")
            else:
                if "asr" in ann.keys():
                    self.subtitle.append(pre_text(ann["asr"]))
                else:
                    self.subtitle.append("")
    

def preprocess_para_retrieval_data(anno_list):
    processed_anno_list = []
    for d in anno_list:
        d["caption"] = " ".join(d.pop("caption"))
        processed_anno_list.append(d)
    return processed_anno_list


class VidTxtRetMCEvalDataset(BaseDataset):
    """For MSRVTT-MC test task"""
    media_type = "video"

    def __init__(self, ann_file, transform, num_frames=4,
                 video_reader_type="decord", sample_type="rand", num_tries=1):
        super(VidTxtRetMCEvalDataset, self).__init__()
        self.anno_list = load_anno(ann_file)
        self.transform = transform
        # video args
        self.num_frames = num_frames
        self.video_reader_type = video_reader_type
        self.video_reader = VIDEO_READER_FUNCS[video_reader_type]
        self.sample_type = sample_type
        self.num_tries = num_tries

    def __len__(self):
        return len(self.anno_list)

    def __getitem__(self, index):
        ann = self.anno_list[index]
        image, index = self.load_and_transform_media_data(index, ann["image"])
        caption = [pre_text(e) for e in ann["caption"]]  # len=5
        answer = ann["answer"]
        return image, caption, answer, ann
    

class VidTxtRetMCNewEvalDataset(BaseDataset):
    """For SSV2-MC and Charades-MC test task"""
    media_type = "video"

    def __init__(self, ann_file, transform, num_frames=4,
                 video_reader_type="decord", sample_type="rand", num_tries=1):
        super(VidTxtRetMCNewEvalDataset, self).__init__()
        self.anno_list = load_anno(ann_file)
        self.transform = transform
        # video args
        self.num_frames = num_frames
        self.video_reader_type = video_reader_type
        self.video_reader = VIDEO_READER_FUNCS[video_reader_type]
        self.sample_type = sample_type
        self.num_tries = num_tries

    def __len__(self):
        return len(self.anno_list)

    def __getitem__(self, index):
        ann = self.anno_list[index]
        image, index = self.load_and_transform_media_data(index, ann["image"])
        option = [pre_text(e) for e in ann["option"]]  # len=174
        answer = ann["answer"]
        if isinstance(answer, list):
            answer = torch.Tensor(answer)
        return image, option, answer, ann


class AudioVidTxtRetEvalDataset(VidTxtRetEvalDataset):
    media_type = "audio_video"

    def __init__(
            self, ann_file, transform, num_frames=4,
            video_reader_type="decord", sample_type="rand", num_tries=1,
            audio_sample_rate=16000, 
            audio_reader_type='torchaudio',
            max_audio_length=10):
        super(AudioVidTxtRetEvalDataset, self).__init__(ann_file, transform, 
            num_frames=num_frames, video_reader_type=video_reader_type,
            sample_type=sample_type, num_tries=num_tries)
        
        self.audio_sample_rate = audio_sample_rate
        self.audio_reader_type = audio_reader_type
        self.max_audio_length = max_audio_length
        self.read_clip_from_video = ann_file.get("read_clip_from_video", False)
        self.read_audio_from_video = ann_file.get("read_audio_from_video", False)
        self.zero_audio_padding_for_video = ann_file.get("zero_audio_padding_for_video", False)

    def __getitem__(self, index):
        ann = self.anno_list[index]
        data_path = {'video': ann["image"]}

        if self.read_clip_from_video:
            raise NotImplementedError("Need to modify load_anno!")
        
        if not self.read_audio_from_video:
            raise NotImplementedError("Need to modify load_anno!")
        
        data_path["read_clip_from_video"] = self.read_clip_from_video
        data_path["read_audio_from_video"] = self.read_audio_from_video

        media, index = self.load_and_transform_media_data(index, data_path)
        audio = media[0]
        if audio is None and self.zero_audio_padding_for_video:
            media[0] = torch.zeros((998, 64), dtype=torch.float32)
        
        return media, index