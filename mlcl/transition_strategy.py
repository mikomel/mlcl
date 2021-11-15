from abc import ABC, abstractmethod

import pytorch_lightning as pl


class TransitionStrategy(ABC):
    @abstractmethod
    def should_transition(self, pl_module: pl.LightningModule) -> bool:
        pass


class EarlyStoppingTransitionStrategy(TransitionStrategy):
    def should_transition(self, pl_module: pl.LightningModule) -> bool:
        early_stop = pl_module.trainer.early_stopping_callback
        is_about_to_stop_early = early_stop.wait_count + 1 >= early_stop.patience
        if is_about_to_stop_early:
            print('EarlyStoppingTransitionStrategy: Recommending transition')
        return is_about_to_stop_early


class AlwaysTrueTransitionStrategy(TransitionStrategy):
    def should_transition(self, pl_module: pl.LightningModule) -> bool:
        if pl_module.trainer.current_epoch > 1:
            print('AlwaysTrueTransitionStrategy: Recommending transition')
            return True
        else:
            return False


class FixedNumEpochsTransitionStrategy(TransitionStrategy):
    def __init__(self, num_epochs: int):
        self.num_epochs = num_epochs

    def should_transition(self, pl_module: pl.LightningModule) -> bool:
        if (pl_module.trainer.current_epoch + 1) % self.num_epochs == 0:
            print('FixedNumEpochsTransitionStrategy: Recommending transition')
            return True
        else:
            return False
