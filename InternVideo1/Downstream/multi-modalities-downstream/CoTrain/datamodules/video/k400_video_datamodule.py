from CoTrain.datasets import K400VideoDataset
from CoTrain.datamodules.image.datamodule_base import BaseDataModule


class K400VideoDataModule(BaseDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def dataset_cls(self):
        return K400VideoDataset

    @property
    def dataset_cls_no_false(self):
        return K400VideoDataset

    @property
    def dataset_name(self):
        return "k400_video"