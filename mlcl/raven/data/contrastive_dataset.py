from typing import Tuple

import numpy as np
import torch

from mlcl.raven.data.auxiliary_dataset import RavenAuxiliaryDataset


class RavenContrastiveDataset(RavenAuxiliaryDataset):
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, torch.Tensor]:
        data = np.load(self.filenames[idx])
        images, target = self._shuffle_answers(data['image'], data['target'])
        meta_matrix = data['meta_matrix']
        images1 = self.augmentor.transform_rpm(images, meta_matrix)
        images2 = self.augmentor.transform_rpm(images, meta_matrix)
        images = np.stack([images1, images2], axis=0)
        images = self._to_tensor(images)
        rules = self.rule_encoder.encode(data)
        return images, target, rules
