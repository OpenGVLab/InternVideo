from CoTrain.datasets import CC12MDataset
from .datamodule_base import BaseDataModule


class CC12MDataModule(BaseDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def dataset_cls(self):
        return CC12MDataset

    @property
    def dataset_name(self):
        return "cc3m"
