import inspect
import random
import sys

import albumentations
import numpy as np


class Roll(albumentations.ImageOnlyTransform):
    def __init__(self, max_vertical_shift=80, max_horizontal_shift=80, always_apply=False, p=0.5):
        super(Roll, self).__init__(always_apply, p)
        self._max_vertical_shift = max_vertical_shift
        self._max_horizontal_shift = max_horizontal_shift

    def apply(self, image, row_shift, col_shift, **params):
        image = np.roll(image, shift=row_shift, axis=0)
        image = np.roll(image, shift=col_shift, axis=1)
        return image

    def get_params(self):
        return {
            'row_shift': random.randint(0, self._max_vertical_shift),
            'col_shift': random.randint(0, self._max_horizontal_shift)
        }

    def get_params_dependent_on_targets(self, params):
        return {}

    def get_transform_init_args_names(self):
        return '_max_vertical_shift', '_max_horizontal_shift'


class HorizontalRoll(Roll):
    def __init__(self, max_shift, **kwargs):
        super(HorizontalRoll, self).__init__(max_vertical_shift=0, max_horizontal_shift=max_shift, **kwargs)


class VerticalRoll(Roll):
    def __init__(self, max_shift, **kwargs):
        super(VerticalRoll, self).__init__(max_vertical_shift=max_shift, max_horizontal_shift=0, **kwargs)


MLCL_AUGMENTATIONS = [
    member[0]
    for member in inspect.getmembers(sys.modules[__name__], inspect.isclass)
]
