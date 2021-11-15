from typing import List, Tuple

import numpy as np
import torch

from mlcl.dataset import DatasetSplit, DEFAULT_DATASET_SPLITS
from mlcl.raven.data.configuration import RavenConfiguration, all_raven_configurations
from mlcl.raven.data.dataset import RavenDataset
from mlcl.raven.data.rule_encoder import RavenRuleEncoder, DenseRavenRuleEncoder
from mlcl.raven.data.augmentor import Augmentor, IDENTITY_AUGMENTOR


class RavenAuxiliaryDataset(RavenDataset):
    def __init__(
            self,
            dataset_root_dir: str = '.',
            configurations: List[RavenConfiguration] = all_raven_configurations,
            splits: List[DatasetSplit] = DEFAULT_DATASET_SPLITS,
            augmentor: Augmentor = IDENTITY_AUGMENTOR,
            rule_encoder: RavenRuleEncoder = DenseRavenRuleEncoder()):
        super(RavenAuxiliaryDataset, self).__init__(dataset_root_dir, configurations, splits, augmentor)
        self.rule_encoder = rule_encoder

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, torch.Tensor]:
        data = np.load(self.filenames[idx])
        images, target = self._shuffle_answers(data['image'], data['target'])
        images = self.augmentor.transform_rpm(images, data['meta_matrix'])
        images = self._to_tensor(images)
        rules = self.rule_encoder.encode(data)
        return images, target, rules
