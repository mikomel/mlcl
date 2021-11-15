from typing import List

import albumentations
import numpy as np


class Augmentor:
    def __init__(
            self,
            p: float = 0.5,
            transforms: List = (),
            position_preserving_transforms: List = ()):
        self.transform = albumentations.Compose(
            transforms,
            additional_targets={f"image{i + 1}": 'image' for i in range(15)},
            p=p)
        self.position_preserving_transform = albumentations.Compose(
            position_preserving_transforms,
            additional_targets={f"image{i + 1}": 'image' for i in range(15)},
            p=p)

    def transform_rpm(self, images: np.array, rules: np.array) -> np.array:
        """
        Applies the same Albumentation transform to all images from RPM.
        :param images: numpy array representing images of RPM with shape (16, width, height)
        :param rules: numpy array representing rules of RPM with shape (4, 12)
        :return: numpy array with augmented RPM images with shape (16, width, height)
        """
        kwargs = {f"image{i + 1}": images[i + 1, :, :] for i in range(15)}
        transform = self.position_preserving_transform if self._has_progression_on_position(rules) else self.transform
        augmented_images = transform(image=images[0, :, :], **kwargs)
        return np.stack([
            augmented_images[f"image{i}"] if i > 0 else augmented_images['image']
            for i in range(16)
        ])

    def _has_progression_on_position(self, rules: np.array) -> bool:
        """ Checks if given RPM has progression relation applied to position. """
        for r in rules:
            if r[4] and r[7]:
                return True
        return False


class IdentityAugmentor(Augmentor):
    def transform_rpm(self, images: np.array, rules: np.array) -> np.array:
        return images


IDENTITY_AUGMENTOR = IdentityAugmentor()
