from abc import ABC, abstractmethod

import numpy as np
import torch


def create_pgm_rule_encoder(name: str) -> "PgmRuleEncoder":
    if name == 'dense':
        return DensePgmRuleEncoder()
    elif name == 'sparse':
        return SparsePgmRuleEncoder()
    else:
        raise ValueError(f"Can't create PgmRuleEncoder with name {name}. Choose one from: {{dense, sparse}}")


class PgmRuleEncoder(ABC):
    @staticmethod
    @abstractmethod
    def encode(data: np.array) -> torch.Tensor:
        pass

    @staticmethod
    @abstractmethod
    def encoding_size() -> int:
        pass


class DensePgmRuleEncoder(PgmRuleEncoder):
    @staticmethod
    def encode(data: np.array) -> torch.Tensor:
        return torch.from_numpy(data['meta_target']).float()

    @staticmethod
    def encoding_size() -> int:
        return 12


class SparsePgmRuleEncoder(PgmRuleEncoder):
    @staticmethod
    def encode(data: np.array) -> torch.Tensor:
        structure = data['relation_structure_encoded']
        rules = torch.zeros(SparsePgmRuleEncoder.encoding_size()).float()
        for i in range(4):
            indices = structure[i, :].nonzero()[0]
            if len(indices) == 3:
                idx = indices[0] * 25 + (indices[1] - 2) * 5 + (indices[2] - 7)
                rules[idx] = 1.0
        return rules

    @staticmethod
    def encoding_size() -> int:
        return 50
