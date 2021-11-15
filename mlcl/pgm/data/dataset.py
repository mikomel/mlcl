import os
import re
from typing import Tuple, List

import numpy as np
import torch
from torch.utils.data import Dataset

from mlcl.dataset import DatasetSplit, DEFAULT_DATASET_SPLITS
from mlcl.raven.data.augmentor import Augmentor, IDENTITY_AUGMENTOR


class PgmDataset(Dataset):
    FILEPATH_PATTERN = re.compile(r"PGM_([\w.]+)_(\w+)_(\d+).npz")

    def __init__(
            self,
            data_dir: str = '.',
            splits: List[DatasetSplit] = DEFAULT_DATASET_SPLITS,
            augmentor: Augmentor = IDENTITY_AUGMENTOR,
            double_downscale_images: bool = False):
        self.filenames = self._list_filenames(data_dir, splits)
        self.augmentor = augmentor
        self.reshape_images = True
        self.double_downscale_images = double_downscale_images

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        data = np.load(self.filenames[idx])
        images, target = data['image'], data['target']
        if self.reshape_images:
            h, w, c = images.shape
            images = np.ascontiguousarray(images.reshape(c, h, w))
        if self.double_downscale_images:
            images = images[:, ::2, ::2]
        images, target = self._shuffle_answers(images, target)
        images = images.astype('float32') / 255.0
        images = self.augmentor.transform_rpm(images, data['relation_structure_encoded'])
        images = self._to_tensor(images)
        return images, target

    @staticmethod
    def _shuffle_answers(images: np.array, target: int) -> Tuple[np.array, int]:
        context_images = images[:8, :, :]
        choice_images = images[8:, :, :]
        indices = list(range(8))
        np.random.shuffle(indices)
        new_target = indices.index(target)
        new_choice_images = choice_images[indices, :, :]
        new_images = np.concatenate((context_images, new_choice_images))
        return new_images, new_target

    @staticmethod
    def _to_tensor(images: np.array) -> torch.Tensor:
        return torch.tensor(images)

    def _list_filenames(self, data_dir: str, splits: List[DatasetSplit]):
        split_names = [s.value for s in splits]
        return [
            os.path.join(data_dir, f)
            for f in os.listdir(data_dir)
            if self._split_filename(f)['dataset_split'] in split_names
        ]

    def _split_filename(self, filename: str):
        match = re.match(self.FILEPATH_PATTERN, filename)
        return {
            'generalisation_split': match.group(1),
            'dataset_split': match.group(2),
            'id': match.group(3)
        }
