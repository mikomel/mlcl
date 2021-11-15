from typing import Tuple

import numpy as np
import torch

from mlcl.pgm.data.auxiliary_dataset import PgmAuxiliaryDataset


class PgmContrastiveDataset(PgmAuxiliaryDataset):
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
        images1 = self.augmentor.transform_rpm(images, data['relation_structure_encoded'])
        images2 = self.augmentor.transform_rpm(images, data['relation_structure_encoded'])
        images = np.stack([images1, images2], axis=0)
        images = self._to_tensor(images)
        rules = self.rule_encoder.encode(data)
        return images, target, rules
