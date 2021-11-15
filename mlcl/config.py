from types import ModuleType
from typing import Iterable, Dict, List

import albumentations
from hydra import compose, initialize
from omegaconf import OmegaConf, DictConfig

import mlcl.augmentation
from mlcl.dataset import DatasetSplit
from mlcl.pgm.data.rule_encoder import create_pgm_rule_encoder
from mlcl.raven.data.configuration import RavenConfiguration
from mlcl.raven.data.rule_encoder import create_raven_rule_encoder


class Config:
    def __init__(self, config_path: str = '../config'):
        self.config_path = config_path
        resolvers = {
            'List': lambda *args: list(args),
            'Tuple': lambda *args: tuple(args),
            'Augmentation': resolve_augmentation,
            'DatasetSplit': lambda x: DatasetSplit[x],
            'RavenConfiguration': lambda x: RavenConfiguration[x],
            'RavenRuleEncoder': create_raven_rule_encoder,
            'PgmRuleEncoder': create_pgm_rule_encoder,
        }
        for name, resolver in resolvers.items():
            OmegaConf.register_new_resolver(name, resolver)

    def compose(self, config_name: str = 'default', overrides: List[str] = ()) -> DictConfig:
        with initialize(config_path=self.config_path):
            return compose(config_name=config_name, overrides=overrides)


def resolve_augmentation(*args):
    class_name = args[0]
    module = mlcl.augmentation if class_name in mlcl.augmentation.MLCL_AUGMENTATIONS else albumentations
    return resolve(module, class_name, *args[1:])


def resolve(module: ModuleType, class_name: str, *args) -> object:
    class_ = getattr(module, class_name)
    kwargs = to_kwargs(*args)
    return class_(**kwargs)


def to_kwargs(*args) -> Dict:
    return {k: v for k, v in pairwise(args)}


def pairwise(iterable: Iterable):
    """s -> (s0, s1), (s2, s3), (s4, s5), ..."""
    iterator = iter(iterable)
    return zip(iterator, iterator)
