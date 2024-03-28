from CoTrain.datasets import ActivityNetDataset
from .datamodule_base import BaseDataModule


class ActivityNetDataModule(BaseDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def dataset_cls(self):
        return ActivityNetDataset

    @property
    def dataset_name(self):
        return "activitynet"
