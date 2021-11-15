from abc import ABC

import pytorch_lightning as pl
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.optim import Optimizer

from mlcl.models.rpm_embedding_model import RPMEmbeddingModel
from mlcl.raven.data.configuration import RavenConfiguration


class RavenModule(pl.LightningModule, ABC):
    def __init__(self, cfg: DictConfig, model: RPMEmbeddingModel):
        super(RavenModule, self).__init__()
        self.save_hyperparameters()
        self.cfg = cfg
        self.model = model

    def configure_optimizers(self):
        optimizer: Optimizer = instantiate(self.cfg.torch.optimizer, params=self.parameters())
        scheduler: object = instantiate(self.cfg.torch.scheduler, optimizer=optimizer)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val/loss'
            }
        }

    def get_configuration(self, split: str, dataloader_idx: int) -> RavenConfiguration:
        if split == 'val':
            return self.cfg.mlcl.raven.dataset.val.configurations[dataloader_idx]
        elif split == 'test':
            return self.cfg.mlcl.raven.dataset.test.configurations[dataloader_idx]
        else:
            raise ValueError(f"No configuration for split: {split}")

    def get_val_configuration(self, dataloader_idx: int) -> RavenConfiguration:
        return self.cfg.mlcl.raven.dataset.val.configurations[dataloader_idx]

    def get_test_configuration(self, dataloader_idx: int) -> RavenConfiguration:
        return self.cfg.mlcl.raven.dataset.test.configurations[dataloader_idx]

    def get_num_val_configurations(self) -> int:
        return len(self.cfg.mlcl.raven.dataset.val.configurations)

    def get_num_test_configurations(self) -> int:
        return len(self.cfg.mlcl.raven.dataset.test.configurations)
