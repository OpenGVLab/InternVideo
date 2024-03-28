import json
from dataset.base_dataset import ImageVideoBaseDataset
from dataset.utils import pre_text, load_anno
from dataset.video_utils import VIDEO_READER_FUNCS
import logging

logger = logging.getLogger(__name__)


class ImageQADataset(ImageVideoBaseDataset):
    media_type = "image"

    def __init__(self, ann_file, transform, eos="[SEP]", mode="train", answer_list=None):
        super(ImageQADataset, self).__init__()
        assert mode in ["train", "eval"]
        self.mode = mode
        self.transform = transform
        self.eos = eos

        self.anno_list = load_anno(ann_file)

        if mode == "eval":
            self.answer_list = json.load(open(answer_list, "r"))

    def __len__(self):
        return len(self.anno_list)

    def get_answers_with_weights(self, raw_answers):
        if isinstance(raw_answers, str):
            raw_answers = [raw_answers]
        answer_weight = {}
        for answer in raw_answers:
            if answer in answer_weight.keys():
                answer_weight[answer] += 1/len(raw_answers)
            else:
                answer_weight[answer] = 1/len(raw_answers)

        answers = list(answer_weight.keys())
        weights = [answer_weight[a] for a in answers]
        answers = [answer + " " + self.eos for answer in answers]
        return answers, weights

    def __getitem__(self, index):
        ann = self.anno_list[index]
        image, index = self.load_and_transform_media_data(index, ann["image"])

        question = pre_text(ann["question"])
        if self.mode == "train":
            answers, weights = self.get_answers_with_weights(ann["answer"])
            return image, question, answers, weights
        else:  # self.mode == "eval":
            question_id = ann["question_id"]
            return image, question, question_id


class VideoQADataset(ImageQADataset):
    media_type = "video"

    def __init__(
            self, ann_file, transform, eos="[SEP]", mode="train", answer_list=None,
            num_frames=4, video_reader_type="decord", sample_type="rand", num_tries=1
    ):
        super(VideoQADataset, self).__init__(
            ann_file, transform, eos, mode, answer_list)
        self.num_frames = num_frames
        self.video_reader_type = video_reader_type
        self.video_reader = VIDEO_READER_FUNCS[video_reader_type]
        self.sample_type = sample_type
        self.num_tries = num_tries
