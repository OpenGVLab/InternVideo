from CoTrain.datasets import MSRVTTDataset
from CoTrain.datamodules.image.datamodule_base import BaseDataModule


class MSRVTTDataModule(BaseDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def dataset_cls(self):
        return MSRVTTDataset

    @property
    def dataset_cls_no_false(self):
        return MSRVTTDataset

    @property
    def dataset_name(self):
        return "msrvtt"
