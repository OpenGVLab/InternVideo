from CoTrain.datasets import HT100MDataset
from CoTrain.datamodules.image.datamodule_base import BaseDataModule


class HT100MDataModule(BaseDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def dataset_cls(self):
        return HT100MDataset

    @property
    def dataset_cls_no_false(self):
        return HT100MDataset

    @property
    def dataset_name(self):
        return "howto100m"
