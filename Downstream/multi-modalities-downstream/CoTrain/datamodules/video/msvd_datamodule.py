from CoTrain.datasets import MSVDDataset
from CoTrain.datamodules.image.datamodule_base import BaseDataModule


class MSVDDataModule(BaseDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def dataset_cls(self):
        return MSVDDataset

    @property
    def dataset_cls_no_false(self):
        return MSVDDataset

    @property
    def dataset_name(self):
        return "msvd"
