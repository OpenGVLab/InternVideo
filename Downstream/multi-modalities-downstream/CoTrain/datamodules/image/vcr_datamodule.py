from CoTrain.datasets import VCRDataset
from .datamodule_base import BaseDataModule


class VCRDataModule(BaseDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def dataset_cls(self):
        return VCRDataset

    @property
    def dataset_name(self):
        return "vcr"
