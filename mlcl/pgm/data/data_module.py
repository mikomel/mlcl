from typing import TypeVar, Generic

import pytorch_lightning as pl
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from mlcl.pgm.data.auxiliary_dataset import PgmAuxiliaryDataset
from mlcl.pgm.data.contrastive_dataset import PgmContrastiveDataset
from mlcl.pgm.data.dataset import PgmDataset

PgmDatasetType = TypeVar('PgmDatasetType', PgmDataset, PgmAuxiliaryDataset, PgmContrastiveDataset)


class PgmDataModule(pl.LightningDataModule, Generic[PgmDatasetType]):
    def __init__(self, cfg: DictConfig):
        super(PgmDataModule, self).__init__()
        self.cfg: DictConfig = cfg
        self.train_dataset: PgmDatasetType = None
        self.val_dataset: PgmDatasetType = None
        self.test_dataset: PgmDatasetType = None

    def setup(self, **kwargs):
        self.train_dataset = instantiate(self.cfg.mlcl.pgm.dataset.train)
        self.val_dataset = instantiate(self.cfg.mlcl.pgm.dataset.val)
        self.test_dataset = instantiate(self.cfg.mlcl.pgm.dataset.test)

    def train_dataloader(self) -> DataLoader:
        return instantiate(self.cfg.torch.data_loader.train, dataset=self.train_dataset, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return instantiate(self.cfg.torch.data_loader.val, dataset=self.val_dataset)

    def test_dataloader(self) -> DataLoader:
        return instantiate(self.cfg.torch.data_loader.test, dataset=self.test_dataset)
