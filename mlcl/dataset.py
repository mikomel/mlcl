from enum import Enum
from typing import List


class DatasetSplit(Enum):
    TRAIN = 'train'
    VAL = 'val'
    TEST = 'test'

    @staticmethod
    def all() -> List["DatasetSplit"]:
        return [e for e in DatasetSplit]


DEFAULT_DATASET_SPLITS = DatasetSplit.all()
