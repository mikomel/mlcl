import torch
from omegaconf import DictConfig
from torch import nn
from torchmetrics import Accuracy

from mlcl.models.copinet import contrast_loss, CoPINet
from mlcl.models.rpm_embedding_model import RPMEmbeddingModel
from mlcl.pgm.module.pgm_module import PgmModule


class PgmSupervisedModule(PgmModule):
    def __init__(self, cfg: DictConfig, model: RPMEmbeddingModel):
        super(PgmSupervisedModule, self).__init__(cfg, model)
        embedding_size = self.model.embedding_size()
        self.score = nn.Linear(embedding_size, 1)
        self.supervised_loss = contrast_loss if isinstance(model, CoPINet) else nn.CrossEntropyLoss()
        self.metrics = nn.ModuleDict({
            'tr': nn.ModuleDict({
                'acc': Accuracy()
            }),
            'val': nn.ModuleDict({
                'acc': Accuracy()
            }),
            'test': nn.ModuleDict({
                'acc': Accuracy()
            })
        })

    def supervised_step(self, candidate_embeddings: torch.Tensor, y: torch.Tensor):
        y_hat = self.score(candidate_embeddings).squeeze()
        loss = self.supervised_loss(y_hat, y)
        return y_hat, loss

    def _step(self, split: str, batch, batch_idx: int) -> torch.Tensor:
        x, y = batch
        candidate_embeddings = self.model(x)
        y_hat, loss = self.supervised_step(candidate_embeddings, y)
        self.log_target_metrics(split, y, y_hat, loss)
        return loss

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        return self._step('tr', batch, batch_idx)

    def validation_step(self, batch, batch_idx: int):
        self._step('val', batch, batch_idx)

    def test_step(self, batch, batch_idx: int):
        self._step('test', batch, batch_idx)

    def log_target_metrics(self, split: str, y: torch.Tensor, y_hat: torch.Tensor, loss: torch.Tensor):
        self.logm(loss, 'loss', split)
        acc = self.metrics[split]['acc'](y_hat, y)
        self.logm(acc, 'acc', split)

    def logm(self, value: torch.Tensor, metric: str, split: str):
        self.log(f"{split}/{metric}", value, on_epoch=True, prog_bar=True, logger=True, add_dataloader_idx=False)

    def logm_type(self, value: torch.Tensor, metric: str, split: str, type: str):
        self.log(f"{split}/{metric}_{type}", value, on_epoch=True, logger=True, add_dataloader_idx=False)
