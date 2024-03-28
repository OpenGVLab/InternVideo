from CoTrain.datasets import YFCC15MDataset
from .datamodule_base import BaseDataModule


class YFCC15MDataModule(BaseDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def dataset_cls(self):
        return YFCC15MDataset

    @property
    def dataset_name(self):
        return "yfcc15m"
