from CoTrain.datasets import HMDB51Dataset
from CoTrain.datamodules.image.datamodule_base import BaseDataModule


class HMDB51DataModule(BaseDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def dataset_cls(self):
        return HMDB51Dataset

    @property
    def dataset_cls_no_false(self):
        return HMDB51Dataset

    @property
    def dataset_name(self):
        return "hmdb51"
