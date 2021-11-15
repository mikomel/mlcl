from typing import Optional

import torch
from omegaconf import DictConfig
from torch import nn

from mlcl.loss import MultiheadSupervisedContrastiveLoss
from mlcl.models.rpm_embedding_model import RPMEmbeddingModel
from mlcl.raven.data.rule_encoder import RavenRuleEncoder, DenseRavenRuleEncoder
from mlcl.raven.module.auxiliary_module import RavenAuxiliaryModule
from mlcl.training_strategy import ThreePhaseTrainableModule


class RavenMLCLModule(RavenAuxiliaryModule, ThreePhaseTrainableModule):
    """
    Multi-label Contrastive Learning (MLCL) module for RAVEN-style datasets.
    This multi-task learning approach combines:
    - supervised training (learning to predict index of the correct answer)
    - auxiliary training (learning to predict the underlying abstract rules)
    - contrastive training (learning to embed RPMs in a semantically-meaningful space)
    """

    def __init__(
            self,
            cfg: DictConfig,
            model: RPMEmbeddingModel,
            auxiliary_loss_scaling: float = 1.0,
            contrastive_loss_scaling: float = 1.0,
            rule_dropout_probability: float = 0.5,
            rule_encoder: RavenRuleEncoder = DenseRavenRuleEncoder(),
            num_contrast_heads: int = 1,
            contrast_embedding_size: int = 128,
            auxiliary_metric_type_name: str = 'auxiliary',
            contrastive_metric_type_name: str = 'contrastive'):
        super(RavenMLCLModule, self).__init__(
            cfg, model, auxiliary_loss_scaling,
            rule_dropout_probability, rule_encoder, auxiliary_metric_type_name)
        self.contrastive_loss_scaling = contrastive_loss_scaling
        self.contrastive_metric_type_name = contrastive_metric_type_name
        embedding_size = self.model.embedding_size()
        self.contrastive_loss = MultiheadSupervisedContrastiveLoss(num_heads=num_contrast_heads)
        self.contrastive_projection = nn.Sequential(
            nn.Linear(embedding_size, embedding_size),
            nn.ReLU(inplace=True),
            nn.Linear(embedding_size, contrast_embedding_size))

    def _step(self, split: str, batch, batch_idx: int, dataloader_idx: Optional[int] = None) -> torch.Tensor:
        x, y, rules = batch
        batch_size, num_views, num_panels, width, height = x.size()
        x = x.view(batch_size * num_views, num_panels, width, height)
        duplicated_y = y.unsqueeze(dim=1).repeat(1, num_views).flatten(0, 1)
        duplicated_rules = rules.unsqueeze(dim=1).repeat(1, num_views, 1).flatten(0, 1)

        candidate_embeddings = self.model(x)
        candidate_embedding_pairs = candidate_embeddings.view(batch_size, num_views, 8, self.model.embedding_size())
        y_hat, supervised_loss = self.supervised_step(candidate_embeddings, duplicated_y)
        rules_hat, auxiliary_loss = self.auxiliary_step(candidate_embeddings, duplicated_rules)
        contrastive_loss = self.contrastive_step(candidate_embedding_pairs, y, rules)
        loss = self.supervised_loss_scaling * supervised_loss \
               + self.auxiliary_loss_scaling * auxiliary_loss \
               + self.contrastive_loss_scaling * contrastive_loss

        self.log_target_metrics(split, duplicated_y, y_hat, supervised_loss, dataloader_idx)
        self.log_auxiliary_metrics(split, duplicated_rules, rules_hat, auxiliary_loss, dataloader_idx)
        self.log_contrastive_metrics(split, contrastive_loss, dataloader_idx)
        self.logm(loss, 'loss', split)
        if dataloader_idx is not None:
            configuration = self.get_val_configuration(dataloader_idx).short_name()
            self.logm_configuration(loss, 'loss', split, configuration)
        return loss

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        return self._step('tr', batch, batch_idx)

    def validation_step(self, batch, batch_idx: int, dataloader_idx: int):
        self._step('val', batch, batch_idx, dataloader_idx)

    def test_step(self, batch, batch_idx: int, dataloader_idx: int):
        self._step('test', batch, batch_idx, dataloader_idx)

    def contrastive_step(
            self,
            candidate_embeddings: torch.Tensor,
            labels: torch.Tensor,
            rules: torch.Tensor) -> torch.Tensor:
        """
        Performs contrastive step with augmented RPMs.
        :param candidate_embeddings: a Tensor with shape (batch_size, num_views, 8, embedding_size)
        """
        device = candidate_embeddings.device
        contrastive_embeddings = self.contrastive_projection(candidate_embeddings)

        batch_size, num_views, num_candidates, embedding_size = contrastive_embeddings.size()
        labels = labels.unsqueeze(1).unsqueeze(2).unsqueeze(3).repeat(1, num_views, 1, embedding_size)
        features = contrastive_embeddings.gather(dim=2, index=labels).squeeze()

        mask = torch.zeros_like(contrastive_embeddings, device=device)
        mask = ~mask.scatter(2, labels, 1.0).bool()
        negatives = contrastive_embeddings[mask].view(batch_size, num_views, 7, embedding_size)
        negatives = negatives.transpose(1, 2).flatten(0, 1)
        loss = self.contrastive_loss(features, negatives, rules)
        return loss

    def on_pre_training(self):
        self.freeze_(self.score)
        self.supervised_loss_scaling = 0

    def on_head_training(self):
        self.freeze_(self.model)
        self.freeze_(self.rule_embedding)
        self.freeze_(self.rule_embedding_summary)
        self.freeze_(self.contrastive_projection)
        self.unfreeze_(self.score)
        self.auxiliary_loss_scaling = 0
        self.contrastive_loss_scaling = 0
        self.supervised_loss_scaling = 1

    def on_fine_tuning(self):
        self.unfreeze_(self.model)

    def log_contrastive_metrics(self, split: str, loss: torch.Tensor, dataloader_idx: Optional[int] = None):
        self.logm_type(loss, 'loss', split, self.contrastive_metric_type_name)
        if dataloader_idx is not None:
            configuration = self.get_configuration(split, dataloader_idx).short_name()
            self.logm_configuration_type(loss, 'loss', split, configuration, self.contrastive_metric_type_name)
