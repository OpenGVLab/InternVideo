from CoTrain.datasets import UCF101Dataset
from CoTrain.datamodules.image.datamodule_base import BaseDataModule


class UCF101DataModule(BaseDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def dataset_cls(self):
        return UCF101Dataset

    @property
    def dataset_cls_no_false(self):
        return UCF101Dataset

    @property
    def dataset_name(self):
        return "ucf101"
