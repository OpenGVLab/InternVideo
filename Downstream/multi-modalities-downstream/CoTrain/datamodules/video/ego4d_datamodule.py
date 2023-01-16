from CoTrain.datasets import Ego4DDataset
from CoTrain.datamodules.image.datamodule_base import BaseDataModule


class Ego4DDataModule(BaseDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def dataset_cls(self):
        return Ego4DDataset

    @property
    def dataset_cls_no_false(self):
        return Ego4DDataset

    @property
    def dataset_name(self):
        return "ego4d"
