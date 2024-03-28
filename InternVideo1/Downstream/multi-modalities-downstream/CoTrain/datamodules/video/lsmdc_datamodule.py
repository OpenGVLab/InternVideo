from CoTrain.datasets import LSMDCDataset
from CoTrain.datamodules.image.datamodule_base import BaseDataModule


class LSMDCDataModule(BaseDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def dataset_cls(self):
        return LSMDCDataset

    @property
    def dataset_cls_no_false(self):
        return LSMDCDataset

    @property
    def dataset_name(self):
        return "lsmdc"
