from typing import TypeVar, Generic, List

import pytorch_lightning as pl
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from mlcl.raven.data.auxiliary_dataset import RavenAuxiliaryDataset
from mlcl.raven.data.configuration import RavenConfiguration
from mlcl.raven.data.contrastive_dataset import RavenContrastiveDataset
from mlcl.raven.data.dataset import RavenDataset

RavenDatasetType = TypeVar('RavenDatasetType', RavenDataset, RavenAuxiliaryDataset, RavenContrastiveDataset)


class RavenDataModule(pl.LightningDataModule, Generic[RavenDatasetType]):
    def __init__(self, cfg: DictConfig):
        super(RavenDataModule, self).__init__()
        self.cfg: DictConfig = cfg
        self.train_dataset: RavenDatasetType = None
        self.val_datasets: List[RavenDatasetType] = None
        self.test_datasets: List[RavenDatasetType] = None

    def setup(self, **kwargs):
        self.train_dataset = instantiate(self.cfg.mlcl.raven.dataset.train)
        self.val_datasets = [
            instantiate(self.cfg.mlcl.raven.dataset.val, configurations=[configuration])
            for configuration in self.cfg.mlcl.raven.dataset.val.configurations
        ]
        self.test_datasets = [
            instantiate(self.cfg.mlcl.raven.dataset.test, configurations=[configuration])
            for configuration in self.cfg.mlcl.raven.dataset.test.configurations
        ]

    def train_dataloader(self) -> DataLoader:
        return instantiate(self.cfg.torch.data_loader.train, dataset=self.train_dataset, shuffle=True)

    def val_dataloader(self) -> List[DataLoader]:
        return [
            instantiate(self.cfg.torch.data_loader.val, dataset=dataset)
            for dataset in self.val_datasets
        ]

    def test_dataloader(self) -> List[DataLoader]:
        return [
            instantiate(self.cfg.torch.data_loader.test, dataset=dataset)
            for dataset in self.test_datasets
        ]

    def get_val_configuration(self, dataloader_idx: int) -> RavenConfiguration:
        return self.cfg.mlcl.raven.dataset.val.configurations[dataloader_idx]

    def get_test_configuration(self, dataloader_idx: int) -> RavenConfiguration:
        return self.cfg.mlcl.raven.dataset.test.configurations[dataloader_idx]
