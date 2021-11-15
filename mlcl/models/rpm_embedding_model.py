from abc import abstractmethod, ABC
from torch import nn


class RPMEmbeddingModel(ABC, nn.Module):
    @abstractmethod
    def embedding_size(self) -> int:
        pass
