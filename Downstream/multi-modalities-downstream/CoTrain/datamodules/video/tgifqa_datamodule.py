from CoTrain.datasets import TGIFQADataset
from CoTrain.datamodules.image.datamodule_base import BaseDataModule


class TGIFQADataModule(BaseDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def dataset_cls(self):
        return TGIFQADataset

    @property
    def dataset_cls_no_false(self):
        return TGIFQADataset

    @property
    def dataset_name(self):
        return "tgifqa"
