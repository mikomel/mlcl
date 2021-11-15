from enum import Enum
from typing import List


class RavenConfiguration(Enum):
    CENTER_SINGLE = 'center_single'
    DISTRIBUTE_FOUR = 'distribute_four'
    DISTRIBUTE_NINE = 'distribute_nine'
    LEFT_CENTER_SINGLE_RIGHT_CENTER_SINGLE = 'left_center_single_right_center_single'
    UP_CENTER_SINGLE_DOWN_CENTER_SINGLE = 'up_center_single_down_center_single'
    IN_CENTER_SINGLE_OUT_CENTER_SINGLE = 'in_center_single_out_center_single'
    IN_DISTRIBUTE_FOUR_OUT_CENTER_SINGLE = 'in_distribute_four_out_center_single'

    @staticmethod
    def all() -> List["RavenConfiguration"]:
        return [e for e in RavenConfiguration]

    def short_name(self) -> str:
        if self == RavenConfiguration.CENTER_SINGLE:
            return 'center'
        elif self == RavenConfiguration.DISTRIBUTE_FOUR:
            return '2x2'
        elif self == RavenConfiguration.DISTRIBUTE_NINE:
            return '3x3'
        elif self == RavenConfiguration.LEFT_CENTER_SINGLE_RIGHT_CENTER_SINGLE:
            return 'left-right'
        elif self == RavenConfiguration.UP_CENTER_SINGLE_DOWN_CENTER_SINGLE:
            return 'up-down'
        elif self == RavenConfiguration.IN_CENTER_SINGLE_OUT_CENTER_SINGLE:
            return 'out-in-center'
        elif self == RavenConfiguration.IN_DISTRIBUTE_FOUR_OUT_CENTER_SINGLE:
            return 'out-in-2x2'


all_raven_configurations = RavenConfiguration.all()
