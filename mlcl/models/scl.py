import torch
from torch import nn

from mlcl.models.conv import ConvBnRelu
from mlcl.models.rpm_embedding_model import RPMEmbeddingModel


class SCL(RPMEmbeddingModel):
    def __init__(self, image_size=80):
        super(SCL, self).__init__()
        self.scattering = Scattering()
        self.conv = nn.Sequential(
            ConvBnRelu(1, 16, kernel_size=3, stride=2, padding=1),
            ConvBnRelu(16, 16, kernel_size=3, padding=1),
            ConvBnRelu(16, 32, kernel_size=3, padding=1),
            ConvBnRelu(32, 32, kernel_size=3, padding=1)
        )
        conv_dimension = 40 * (image_size // 80) * 40 * (image_size // 80)
        self.conv_projection = nn.Sequential(
            nn.Linear(conv_dimension, 80),
            nn.ReLU(inplace=True)
        )
        self.ff_object = FeedForwardResidualBlock(80)
        self.attribute_network = nn.Sequential(
            nn.Linear(32 * 8, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 8)
        )
        self.ff_attribute = FeedForwardResidualBlock(80)
        self.relation_network = nn.Sequential(
            nn.Linear(9, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 5)
        )
        self.ff_relation = FeedForwardResidualBlock(5 * 80)

    def embedding_size(self) -> int:
        return 5 * 80

    def forward(self, x: torch.Tensor):
        batch_size, num_panels, height, width = x.size()

        x = x.view(batch_size * num_panels, 1, height, width)
        x = self.conv(x)
        x = x.view(batch_size, num_panels, 32, -1)
        x = self.conv_projection(x)
        x = self.ff_object(x)

        x = self.scattering(x, num_groups=10)
        x = self.attribute_network(x)
        x = x.view(batch_size, num_panels, 10 * 8)
        x = self.ff_attribute(x)

        x = torch.cat([
            x[:, :8, :].unsqueeze(dim=1).repeat(1, 8, 1, 1),
            x[:, 8:, :].unsqueeze(dim=2)
        ], dim=2)

        x = self.scattering(x, num_groups=80)
        x = self.relation_network(x)
        x = x.view(batch_size, 8, 80 * 5)
        x = self.ff_relation(x)
        return x


class FeedForwardResidualBlock(nn.Module):
    def __init__(self, dim, expansion_multiplier=1):
        super(FeedForwardResidualBlock, self).__init__()
        self._projection = nn.Sequential(
            nn.Linear(dim, dim * expansion_multiplier),
            nn.ReLU(inplace=True),
            nn.LayerNorm(dim * expansion_multiplier),
            nn.Linear(dim * expansion_multiplier, dim)
        )

    def forward(self, x: torch.Tensor):
        return x + self._projection(x)


class Scattering(nn.Module):
    def forward(self, x, num_groups):
        """
        :param x: a Tensor with rank >= 3 and last dimension divisible by number of groups
        :param num_groups: number of groups
        """
        shape_1 = x.shape[:-1] + (num_groups,) + (x.shape[-1] // num_groups,)
        x = x.view(shape_1)
        x = x.transpose(-3, -2).contiguous()
        return x.flatten(start_dim=-2)
