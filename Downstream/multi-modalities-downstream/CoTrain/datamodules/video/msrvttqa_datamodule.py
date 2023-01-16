from CoTrain.datasets import MSRVTTQADataset
from CoTrain.datamodules.image.datamodule_base import BaseDataModule
from collections import defaultdict


class MSRVTTQADataModule(BaseDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def dataset_cls(self):
        return MSRVTTQADataset

    @property
    def dataset_name(self):
        return "msrvttqa"

    def setup(self, stage):
        super().setup(stage)
        self.answer2id = self.train_dataset.ans_lab_dict
        sorted_a2i = sorted(self.answer2id.items(), key=lambda x: x[1])
        self.num_class = max(self.answer2id.values()) + 1
        self.id2answer = defaultdict(lambda: "unknown")
        for k, v in sorted_a2i:
            self.id2answer[v] = k
