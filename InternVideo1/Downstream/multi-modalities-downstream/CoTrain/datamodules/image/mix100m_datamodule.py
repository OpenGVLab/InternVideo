from CoTrain.datasets import MIX100MDataset
from .datamodule_base import BaseDataModule


class MIX100MDataModule(BaseDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def dataset_cls(self):
        return MIX100MDataset

    @property
    def dataset_name(self):
        return "mix100m"
