from CoTrain.datasets import LAION400MDataset
from .datamodule_base import BaseDataModule


class LAION400MDataModule(BaseDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def dataset_cls(self):
        return LAION400MDataset

    @property
    def dataset_name(self):
        return "laion400m"
