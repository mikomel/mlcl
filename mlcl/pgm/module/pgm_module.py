from abc import ABC

import pytorch_lightning as pl
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.optim import Optimizer

from mlcl.models.rpm_embedding_model import RPMEmbeddingModel


class PgmModule(pl.LightningModule, ABC):
    def __init__(self, cfg: DictConfig, model: RPMEmbeddingModel):
        super(PgmModule, self).__init__()
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
