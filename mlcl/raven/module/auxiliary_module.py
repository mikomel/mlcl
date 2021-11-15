from typing import Tuple, Optional

import torch
from omegaconf import DictConfig
from torch import nn
from torch.nn import functional as F
from torchmetrics import Accuracy

from mlcl.models.rpm_embedding_model import RPMEmbeddingModel
from mlcl.raven.data.rule_encoder import RavenRuleEncoder, DenseRavenRuleEncoder
from mlcl.raven.module.supervised_module import RavenSupervisedModule


class RavenAuxiliaryModule(RavenSupervisedModule):
    def __init__(
            self,
            cfg: DictConfig,
            model: RPMEmbeddingModel,
            auxiliary_loss_scaling: float = 1.0,
            rule_dropout_probability: float = 0.5,
            rule_encoder: RavenRuleEncoder = DenseRavenRuleEncoder(),
            metric_type_name: str = 'auxiliary'):
        super(RavenAuxiliaryModule, self).__init__(cfg, model)
        self.supervised_loss_scaling = 1
        self.auxiliary_loss_scaling = auxiliary_loss_scaling
        self.metric_type_name = metric_type_name
        embedding_size = self.model.embedding_size()
        self.rule_embedding = nn.Sequential(
            nn.Linear(embedding_size, embedding_size),
            nn.ReLU(inplace=True),
            nn.Dropout(rule_dropout_probability),
            nn.Linear(embedding_size, rule_encoder.encoding_size())
        )
        self.rule_embedding_summary = nn.Linear(8 * rule_encoder.encoding_size(), rule_encoder.encoding_size())
        self.metrics = nn.ModuleDict({
            'tr': nn.ModuleDict({
                'acc': Accuracy(),
                'rule_acc': Accuracy()
            }),
            'val': nn.ModuleDict({
                'acc': Accuracy(),
                'rule_acc': Accuracy(),
                'accs': nn.ModuleList([Accuracy() for _ in range(self.get_num_val_configurations())]),
                'rule_accs': nn.ModuleList([Accuracy() for _ in range(self.get_num_val_configurations())])
            }),
            'test': nn.ModuleDict({
                'acc': Accuracy(),
                'rule_acc': Accuracy(),
                'accs': nn.ModuleList([Accuracy() for _ in range(self.get_num_test_configurations())]),
                'rule_accs': nn.ModuleList([Accuracy() for _ in range(self.get_num_test_configurations())])
            })
        })

    def auxiliary_step(
            self,
            candidate_embeddings: torch.Tensor,
            rules: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        rules_hat = self.rule_embedding(candidate_embeddings)
        rules_hat = rules_hat.squeeze().flatten(-2, -1)
        rules_hat = self.rule_embedding_summary(rules_hat)
        loss = F.binary_cross_entropy_with_logits(rules_hat, rules)
        return rules_hat, loss

    def _step(self, split: str, batch, batch_idx: int, dataloader_idx: Optional[int] = None) -> torch.Tensor:
        x, y, rules = batch
        candidate_embeddings = self.model(x)
        y_hat, supervised_loss = self.supervised_step(candidate_embeddings, y)
        rules_hat, auxiliary_loss = self.auxiliary_step(candidate_embeddings, rules)
        loss = supervised_loss + self.auxiliary_loss_scaling * auxiliary_loss
        self.log_target_metrics(split, y, y_hat, supervised_loss, dataloader_idx)
        self.log_auxiliary_metrics(split, rules, rules_hat, auxiliary_loss, dataloader_idx)
        self.logm(loss, 'loss', split)
        if dataloader_idx is not None:
            configuration = self.get_val_configuration(dataloader_idx).short_name()
            self.logm_configuration(loss, 'loss', split, configuration)
        return loss

    def training_step(self, batch, batch_idx):
        return self._step('tr', batch, batch_idx)

    def validation_step(self, batch, batch_idx, dataloader_idx):
        self._step('val', batch, batch_idx, dataloader_idx)

    def test_step(self, batch, batch_idx, dataloader_idx):
        self._step('test', batch, batch_idx, dataloader_idx)

    def log_target_metrics(
            self, split: str, y: torch.Tensor, y_hat: torch.Tensor,
            loss: torch.Tensor, dataloader_idx: Optional[int] = None):
        self.logm_type(loss, 'loss', split, 'supervised')
        acc = self.metrics[split]['acc'](y_hat, y)
        self.logm(acc, 'acc', split)
        if dataloader_idx is not None:
            configuration = self.get_configuration(split, dataloader_idx).short_name()
            self.logm_configuration_type(loss, 'loss', split, configuration, 'supervised')
            acc_configuration = self.metrics[split]['accs'][dataloader_idx](y_hat, y)
            self.logm_configuration(acc_configuration, 'acc', split, configuration)

    def log_auxiliary_metrics(
            self, split: str, rules: torch.Tensor, rules_hat: torch.Tensor,
            loss: torch.Tensor, dataloader_idx: Optional[int] = None):
        self.logm_type(loss, 'loss', split, self.metric_type_name)
        acc_rule = self.metrics[split]['rule_acc'](rules_hat.sigmoid(), rules.int())
        self.logm_type(acc_rule, 'acc_rule', split, self.metric_type_name)
        if dataloader_idx is not None:
            configuration = self.get_configuration(split, dataloader_idx).short_name()
            self.logm_configuration_type(loss, 'loss', split, configuration, self.metric_type_name)
            acc_rule_configuration = self.metrics[split]['rule_accs'][dataloader_idx](rules_hat.sigmoid(), rules.int())
            self.logm_configuration_type(acc_rule_configuration, 'acc_rule', split, configuration, self.metric_type_name)
