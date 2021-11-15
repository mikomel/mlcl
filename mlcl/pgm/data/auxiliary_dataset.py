from typing import List, Tuple

import numpy as np
import torch

from mlcl.dataset import DatasetSplit, DEFAULT_DATASET_SPLITS
from mlcl.pgm.data.augmentor import Augmentor, IDENTITY_AUGMENTOR
from mlcl.pgm.data.dataset import PgmDataset
from mlcl.pgm.data.rule_encoder import PgmRuleEncoder, DensePgmRuleEncoder


class PgmAuxiliaryDataset(PgmDataset):
    def __init__(
            self,
            data_dir: str = '.',
            splits: List[DatasetSplit] = DEFAULT_DATASET_SPLITS,
            augmentor: Augmentor = IDENTITY_AUGMENTOR,
            double_downscale_images: bool = False,
            rule_encoder: PgmRuleEncoder = DensePgmRuleEncoder()):
        super(PgmAuxiliaryDataset, self).__init__(data_dir, splits, augmentor, double_downscale_images)
        self.rule_encoder = rule_encoder

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, torch.Tensor]:
        data = np.load(self.filenames[idx])
        images, target = data['image'], data['target']
        if self.reshape_images:
            h, w, c = images.shape
            images = images.reshape(c, h, w)
        if self.double_downscale_images:
            images = images[:, ::2, ::2]
        images, target = self._shuffle_answers(images, target)
        images = images.astype('float32') / 255.0
        images = self.augmentor.transform_rpm(images, data['relation_structure_encoded'])
        images = self._to_tensor(images)
        rules = self.rule_encoder.encode(data)
        return images, target, rules
