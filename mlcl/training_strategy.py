from abc import ABC
from enum import Enum
from typing import Union, Dict, Any, List

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import BaseFinetuning
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from torch import nn
from torch.optim import Optimizer

from mlcl.transition_strategy import TransitionStrategy


class TrainingPhase(Enum):
    PRE_TRAINING = 'pre-training'
    HEAD_TRAINING = 'head training'
    FINE_TUNING = 'fine-tuning'

    def next(self) -> 'TrainingPhase':
        if self == TrainingPhase.PRE_TRAINING:
            return TrainingPhase.HEAD_TRAINING
        elif self == TrainingPhase.HEAD_TRAINING:
            return TrainingPhase.FINE_TUNING
        else:
            raise ValueError(f"{self} doesn't have a successor.")


class ThreePhaseTrainingStrategy(BaseFinetuning):
    """
    Training strategy for Multi-label Contrastive Learning (MLCL). The training will be partitioned into 3 phases:
    1. pre-training: auxiliary training + contrastive training
    2. training attached classification head: supervised training with frozen encoder network
    3. fine-tuning: end-to-end supervised training
    """

    def __init__(self, transition_strategy: TransitionStrategy):
        super(ThreePhaseTrainingStrategy, self).__init__()
        self.training_phase = TrainingPhase.PRE_TRAINING
        self.transition_strategy = transition_strategy
        self.initial_lr: float = 3e-4

    def on_save_checkpoint(
            self,
            trainer: pl.Trainer,
            pl_module: pl.LightningModule,
            checkpoint: Dict[str, Any]
    ) -> Dict[str, Any]:
        return {
            'internal_optimizer_metadata': self._internal_optimizer_metadata,
            'training_phase': self.training_phase,
        }

    def on_load_checkpoint(
            self,
            trainer: pl.Trainer,
            pl_module: pl.LightningModule,
            callback_state: Dict[int, List[Dict[str, Any]]]
    ) -> None:
        self.training_phase = callback_state['training_phase']
        super().on_load_checkpoint(trainer, pl_module, callback_state['internal_optimizer_metadata'])

    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        if isinstance(pl_module, ThreePhaseTrainableModule) and isinstance(pl_module, pl.LightningModule):
            return super().on_fit_start(trainer, pl_module)
        raise MisconfigurationException('The LightningModule should be also an instance of ThreePhaseTrainableModule')

    def freeze_before_training(self, pl_module: Union['ThreePhaseTrainableModule', pl.LightningModule]):
        pl_module.on_pre_training()

    def finetune_function(
            self,
            pl_module: Union['ThreePhaseTrainableModule', pl.LightningModule],
            epoch: int,
            optimizer: Optimizer,
            opt_idx: int):
        if self.transition_strategy.should_transition(pl_module):
            if TrainingPhase.PRE_TRAINING == self.training_phase:
                print('ThreePhaseTrainingStrategy: transitioning from TrainingPhase.PRE_TRAINING to TrainingPhase.HEAD_TRAINING')
                self.training_phase = TrainingPhase.HEAD_TRAINING
                pl_module.on_head_training()
                self.transition_to_next_phase(pl_module.trainer)
            elif TrainingPhase.HEAD_TRAINING == self.training_phase:
                print('ThreePhaseTrainingStrategy: transitioning from TrainingPhase.HEAD_TRAINING to TrainingPhase.FINE_TUNING')
                self.training_phase = TrainingPhase.FINE_TUNING
                pl_module.on_fine_tuning()
                self.transition_to_next_phase(pl_module.trainer)

    def transition_to_next_phase(self, trainer: pl.Trainer):
        torch_inf = torch.tensor(np.Inf)
        trainer.early_stopping_callback.best_score = torch_inf if trainer.early_stopping_callback.monitor_op == torch.lt else -torch_inf
        trainer.early_stopping_callback.wait_count = 0
        print(f"ThreePhaseTrainingStrategy: changing lr from {trainer.optimizers[0].param_groups[0]['lr']:.8f} to {self.initial_lr:.8f}")
        trainer.lr_schedulers[0]['scheduler']._reset()
        for i, param_group in enumerate(trainer.optimizers[0].param_groups):
            param_group['lr'] = self.initial_lr


class ThreePhaseTrainableModule(ABC):
    def on_pre_training(self):
        pass

    def on_head_training(self):
        pass

    def on_fine_tuning(self):
        pass

    @staticmethod
    def freeze_(module: nn.Module):
        module.requires_grad_(False)

    @staticmethod
    def unfreeze_(module: nn.Module):
        module.requires_grad_(True)
