from CoTrain.datasets import WEBVIDDataset
from CoTrain.datamodules.image.datamodule_base import BaseDataModule


class WEBVIDDataModule(BaseDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def dataset_cls(self):
        return WEBVIDDataset

    @property
    def dataset_cls_no_false(self):
        return WEBVIDDataset

    @property
    def dataset_name(self):
        return "webvid"
