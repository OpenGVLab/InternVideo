from CoTrain.datasets import EGO4DChoiceDataset
from CoTrain.datamodules.image.datamodule_base import BaseDataModule


class EGO4DChoiceDataModule(BaseDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def dataset_cls(self):
        return EGO4DChoiceDataset

    @property
    def dataset_cls_no_false(self):
        return EGO4DChoiceDataset

    @property
    def dataset_name(self):
        return "ego4d_choice"
