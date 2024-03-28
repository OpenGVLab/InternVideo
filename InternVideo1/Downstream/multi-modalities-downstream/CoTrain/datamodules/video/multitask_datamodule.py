import functools

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from torch.utils.data.dataset import ConcatDataset
from torch.utils.data.distributed import DistributedSampler
from pytorch_lightning.trainer.supporters import CombinedLoader

from CoTrain.datamodules import _datamodules


class MTDataModule(LightningDataModule):
    def __init__(self, _config, dist=False):
        video_datamodule_keys = _config["video_datasets"]
        image_datamodule_keys = _config["image_datasets"]
        self.num_video_datasets = len(video_datamodule_keys)
        self.num_image_datasets = len(image_datamodule_keys)

        assert self.num_video_datasets > 0 or self.num_image_datasets > 0

        super().__init__()
        if self.num_video_datasets > 0:
            self.video_dm_keys = video_datamodule_keys
            self.video_dm_dicts = {key: _datamodules[key](_config) for key in video_datamodule_keys}
            self.video_dms = [v for k, v in self.video_dm_dicts.items()]
            self.video_batch_size = self.video_dms[0].batch_size
            self.video_vocab_size = self.video_dms[0].vocab_size
            self.video_num_workers = self.video_dms[0].num_workers
        if self.num_image_datasets:
            self.image_dm_keys = image_datamodule_keys
            self.image_dm_dicts = {key: _datamodules[key](_config) for key in image_datamodule_keys}
            self.image_dms = [v for k, v in self.image_dm_dicts.items()]
            self.image_batch_size = self.image_dms[0].batch_size * _config["image_data_mult"]
            self.image_vocab_size = self.image_dms[0].vocab_size
            self.image_num_workers = self.image_dms[0].num_workers
        self.dist = dist

        # We add extra val datamodules so that we can use different dataset in train and val
        # We assume all val datasets are video datasets
        self.val_dm_keys = _config["val_datasets"]
        self.val_dm_dicts = {
            key: _datamodules[key](_config) for key in self.val_dm_keys
        }
        self.val_dms = [v for k, v in self.val_dm_dicts.items()]

        self.pin_memory = False

    def prepare_data(self):
        if self.num_video_datasets:
            for dm in self.video_dms:
                dm.prepare_data()
        if self.num_image_datasets:
            for dm in self.image_dms:
                dm.prepare_data()
        for dm in self.val_dms:
            dm.prepare_data()

    def setup(self, stage):
        if self.num_video_datasets:
            for dm in self.video_dms:
                dm.setup(stage)
        if self.num_image_datasets:
            for dm in self.image_dms:
                dm.setup(stage)
        for dm in self.val_dms:
            dm.setup(stage)

        if self.num_video_datasets:
            self.video_train_dataset = ConcatDataset([dm.train_dataset for dm in self.video_dms])
            self.video_val_dataset = ConcatDataset([dm.val_dataset for dm in self.video_dms])
            self.video_test_dataset = ConcatDataset([dm.test_dataset for dm in self.video_dms])

        if self.num_image_datasets:
            self.image_train_dataset = ConcatDataset([dm.train_dataset for dm in self.image_dms])
            self.image_val_dataset = ConcatDataset([dm.val_dataset for dm in self.image_dms])
            self.image_test_dataset = ConcatDataset([dm.test_dataset for dm in self.image_dms])

        if len(self.val_dms) == 0:
            self.val_dataset = None
        else:
            self.val_dataset = ConcatDataset([dm.val_dataset for dm in self.val_dms])

        if len(self.video_dms) > 0:
            self.tokenizer = self.video_dms[0].tokenizer
        else:
            self.tokenizer = self.image_dms[0].tokenizer

        if self.num_video_datasets:
            self.video_collate = functools.partial(
                self.video_dms[0].train_dataset.collate, mlm_collator=self.video_dms[0].mlm_collator,
            )
        if self.num_image_datasets:
            self.image_collate = functools.partial(
                self.image_dms[0].train_dataset.collate, mlm_collator=self.image_dms[0].mlm_collator,
            )

        if self.dist:
            if self.num_video_datasets:
                self.video_train_sampler = DistributedSampler(self.video_train_dataset, shuffle=False)
                self.video_val_sampler = DistributedSampler(self.video_val_dataset, shuffle=False)
                self.video_test_sampler = DistributedSampler(self.video_test_dataset, shuffle=False)
            if self.num_image_datasets:
                self.image_train_sampler = DistributedSampler(self.image_train_dataset, shuffle=False)
                self.image_val_sampler = DistributedSampler(self.image_val_dataset, shuffle=False)
                self.image_test_sampler = DistributedSampler(self.image_test_dataset, shuffle=False)
            if self.val_dataset is not None:
                self.val_sampler = DistributedSampler(self.val_dataset, shuffle=False)
        else:
            self.video_train_sampler = None
            self.video_val_sampler = None
            self.video_test_sampler = None
            self.image_train_sampler = None
            self.image_val_sampler = None
            self.image_test_sampler = None

    def train_dataloader(self):
        if self.num_video_datasets:
            video_loader = DataLoader(
                self.video_train_dataset,
                batch_size=self.video_batch_size,
                sampler=self.video_train_sampler,
                num_workers=self.video_num_workers,
                pin_memory=self.pin_memory,
                collate_fn=self.video_collate,
            )
        if self.num_image_datasets:
            image_loader = DataLoader(
                self.image_train_dataset,
                batch_size=self.image_batch_size,
                sampler=self.image_train_sampler,
                num_workers=self.image_num_workers,
                pin_memory=self.pin_memory,
                collate_fn=self.image_collate,
            )
        if self.num_video_datasets and self.num_image_datasets:
            loaders = {"v": video_loader, "i": image_loader}
            combined_loader = CombinedLoader(loaders, mode="min_size")  # "min_size" / "max_size_cycle",
        else:
            if self.num_video_datasets:
                combined_loader = video_loader
            else:
                combined_loader = image_loader
        return combined_loader

    def val_dataloader(self, batch_size=None):
        # Skip all other datasets if we have different val datasets
        if self.val_dataset is not None:
            return DataLoader(
                self.val_dataset,
                # batch_size=batch_size if batch_size is not None else self.video_batch_size * 4,
                batch_size=batch_size if batch_size is not None else self.video_batch_size,
                sampler=self.val_sampler,
                num_workers=self.video_num_workers // 2,
                pin_memory=self.pin_memory,
                collate_fn=self.video_collate,
            )
        if self.num_video_datasets:
            video_loader = DataLoader(
                self.video_val_dataset,
                batch_size=batch_size if batch_size is not None else self.video_batch_size,
                sampler=self.video_val_sampler,
                num_workers=self.video_num_workers,
                pin_memory=self.pin_memory,
                collate_fn=self.video_collate,
            )
        if self.num_image_datasets:
            image_loader = DataLoader(
                self.image_val_dataset,
                batch_size=batch_size if batch_size is not None else self.image_batch_size,
                sampler=self.image_val_sampler,
                num_workers=self.image_num_workers,
                pin_memory=self.pin_memory,
                collate_fn=self.image_collate,
            )
        if self.num_video_datasets and self.num_image_datasets:
            loaders = {"v": video_loader, "i": image_loader}
            combined_loader = CombinedLoader(loaders, mode="min_size")  # min_size / max_size_cycle
        else:
            if self.num_video_datasets:
                combined_loader = video_loader
            else:
                combined_loader = image_loader
        return combined_loader


    def test_dataloader(self):
        if self.num_video_datasets:
            video_loader = DataLoader(
                self.video_test_dataset,
                batch_size=self.video_batch_size,
                sampler=self.video_test_sampler,
                num_workers=self.video_num_workers,
                pin_memory=self.pin_memory,
                collate_fn=self.video_collate,
            )
        if self.num_image_datasets:
            image_loader = DataLoader(
                self.image_test_dataset,
                batch_size=self.image_batch_size,
                sampler=self.image_test_sampler,
                num_workers=self.image_num_workers,
                pin_memory=self.pin_memory,
                collate_fn=self.image_collate,
            )
        if self.num_video_datasets and self.num_image_datasets:
            loaders = {"v": video_loader, "i": image_loader}
            combined_loader = CombinedLoader(loaders, mode="min_size")  # min_size / max_size_cycle
        else:
            if self.num_video_datasets:
                combined_loader = video_loader
            else:
                combined_loader = image_loader
        return combined_loader

