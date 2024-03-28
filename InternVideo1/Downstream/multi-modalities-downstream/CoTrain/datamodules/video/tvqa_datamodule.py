from CoTrain.datasets import TVQADataset
from CoTrain.datamodules.image.datamodule_base import BaseDataModule


class TVQADataModule(BaseDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def dataset_cls(self):
        return TVQADataset

    @property
    def dataset_cls_no_false(self):
        return TVQADataset

    @property
    def dataset_name(self):
        return "tvqa"
