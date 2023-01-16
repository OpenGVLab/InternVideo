from CoTrain.datasets import DIDEMODataset
from CoTrain.datamodules.image.datamodule_base import BaseDataModule


class DIDEMODataModule(BaseDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def dataset_cls(self):
        return DIDEMODataset

    @property
    def dataset_cls_no_false(self):
        return DIDEMODataset

    @property
    def dataset_name(self):
        return "didemo"
