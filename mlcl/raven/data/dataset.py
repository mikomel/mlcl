import glob
import os
import re
from typing import Tuple, List

import numpy as np
import torch
from torch.utils.data import Dataset

from mlcl.dataset import DatasetSplit, DEFAULT_DATASET_SPLITS
from mlcl.raven.data.configuration import RavenConfiguration, all_raven_configurations
from mlcl.raven.data.augmentor import Augmentor, IDENTITY_AUGMENTOR


class RavenDataset(Dataset):
    FILEPATH_PATTERN = re.compile(r".*/(\w+)/RAVEN_(\d+)_(\w+).npz")

    def __init__(
            self,
            dataset_root_dir: str = '.',
            configurations: List[RavenConfiguration] = all_raven_configurations,
            splits: List[DatasetSplit] = DEFAULT_DATASET_SPLITS,
            augmentor: Augmentor = IDENTITY_AUGMENTOR):
        self.dataset_root_dir = dataset_root_dir
        self.configuration_names = [c.value for c in configurations]
        self.split_names = [s.value for s in splits]
        self.filenames = self._list_filenames()
        self.augmentor = augmentor

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        data = np.load(self.filenames[idx])
        images, target = self._shuffle_answers(data['image'], data['target'])
        images = self.augmentor.transform_rpm(images, data['meta_matrix'])
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
        images = images.astype('float32') / 255.0
        return torch.tensor(images)

    def _list_filenames(self):
        filenames = []
        for configuration in self.configuration_names:
            filename_pattern = os.path.join(self.dataset_root_dir, configuration, '*.npz')
            filename_pattern = os.path.expanduser(filename_pattern)
            configuration_filenames = glob.glob(filename_pattern)
            for f in configuration_filenames:
                if self._should_contain_filename(f):
                    filenames.append(f)
        return filenames

    def _should_contain_filename(self, filename: str):
        filename = self._split_filename(filename)
        return filename['configuration'] in self.configuration_names and filename['split'] in self.split_names

    def _split_filename(self, filename: str):
        match = re.match(self.FILEPATH_PATTERN, filename)
        return {'configuration': match.group(1), 'id': match.group(2), 'split': match.group(3)}
