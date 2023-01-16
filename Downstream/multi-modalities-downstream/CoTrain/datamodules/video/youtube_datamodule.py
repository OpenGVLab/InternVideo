from CoTrain.datasets import YOUTUBEDataset
from CoTrain.datamodules.image.datamodule_base import BaseDataModule


class YOUTUBEDataModule(BaseDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def dataset_cls(self):
        return YOUTUBEDataset

    @property
    def dataset_cls_no_false(self):
        return YOUTUBEDataset

    @property
    def dataset_name(self):
        return "youtube"
