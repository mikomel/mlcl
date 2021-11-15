from abc import ABC, abstractmethod

import numpy as np
import torch


def create_raven_rule_encoder(name: str) -> "RavenRuleEncoder":
    if name == 'dense':
        return DenseRavenRuleEncoder()
    elif name == 'sparse':
        return SparseRavenRuleEncoder()
    else:
        raise ValueError(f"Can't create RavenRuleEncoder with name {name}. Choose one from: {{dense, sparse}}")


class RavenRuleEncoder(ABC):
    @staticmethod
    @abstractmethod
    def encode(data: np.array) -> torch.Tensor:
        pass

    @staticmethod
    @abstractmethod
    def encoding_size() -> int:
        pass


class DenseRavenRuleEncoder(RavenRuleEncoder):
    @staticmethod
    def encode(data: np.array) -> torch.Tensor:
        return torch.from_numpy(data['meta_target']).float()

    @staticmethod
    def encoding_size() -> int:
        return 9


class SparseRavenRuleEncoder(RavenRuleEncoder):
    @staticmethod
    def encode(data: np.array) -> torch.Tensor:
        meta_matrix = data['meta_matrix']
        rules = torch.zeros(SparseRavenRuleEncoder.encoding_size())
        for row in meta_matrix[:4]:
            rule = row[:4].argmax()
            attributes = np.where(row[4:] == 1)[0]
            rules[4 * attributes + rule] = 1
        for row in meta_matrix[4:]:
            rule = row[:4].argmax()
            attributes = np.where(row[4:] == 1)[0]
            rules[4 * 5 + attributes + rule * 5] = 1
        return rules

    @staticmethod
    def encoding_size() -> int:
        return 40
