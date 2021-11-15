import torch
from torch import nn
from torchvision.models import resnet18

from mlcl.models.linear_bn_relu import DeepLinearBnRelu
from mlcl.models.rpm_embedding_model import RPMEmbeddingModel


class SRAN(RPMEmbeddingModel):
    def __init__(self):
        super(SRAN, self).__init__()
        resnet_embedding_size = 512

        self.cell_cnn = resnet18(pretrained=False)
        self.cell_cnn.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.cell_cnn.fc = nn.Identity()

        self.ind_cnn = resnet18(pretrained=False)
        self.ind_cnn.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.ind_cnn.fc = nn.Identity()

        self.eco_cnn = resnet18(pretrained=False)
        self.eco_cnn.conv1 = nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.eco_cnn.fc = nn.Identity()

        cell_input_size = 3 * resnet_embedding_size
        cell_output_size = resnet_embedding_size
        ind_input_size = cell_output_size + resnet_embedding_size + cell_output_size + resnet_embedding_size
        ind_output_size = resnet_embedding_size
        eco_input_size = ind_output_size + resnet_embedding_size
        eco_output_size = resnet_embedding_size
        self.eco_output_size = eco_output_size
        self.cell_gated_embedding_fusion = DeepLinearBnRelu(depth=2, in_dim=cell_input_size, out_dim=cell_output_size, change_dim_first=False)
        self.ind_gated_embedding_fusion = DeepLinearBnRelu(depth=2, in_dim=ind_input_size, out_dim=ind_output_size, change_dim_first=False)
        self.eco_gated_embedding_fusion = DeepLinearBnRelu(depth=3, in_dim=eco_input_size, out_dim=eco_output_size, change_dim_first=False)
        self.embedding = DeepLinearBnRelu(depth=2, in_dim=2 * eco_output_size, out_dim=2 * eco_output_size, change_dim_first=False)

    def embedding_size(self) -> int:
        return 2 * self.eco_output_size

    def forward(self, x):
        batch_size, num_panels, height, width = x.size()

        # Cell-wise hierarchy
        x_cell = x.view(batch_size * num_panels, 1, height, width)
        x_cell = self.cell_cnn(x_cell)
        x_cell = x_cell.view(batch_size, num_panels, -1)

        cell_r_1 = x_cell[:, :3, :].reshape(batch_size, -1)
        cell_r_2 = x_cell[:, 3:6, :].reshape(batch_size, -1)
        cell_r_1 = self.cell_gated_embedding_fusion(cell_r_1)
        cell_r_2 = self.cell_gated_embedding_fusion(cell_r_2)

        # Individual-wise hierarchy
        row_1 = x[:, :3, :, :]
        row_2 = x[:, 3:6, :, :]
        ind_r_1 = self.ind_cnn(row_1).view(batch_size, -1)
        ind_r_2 = self.ind_cnn(row_2).view(batch_size, -1)

        ind_r_1_2 = torch.cat([cell_r_1, ind_r_1, cell_r_2, ind_r_2], dim=1)
        ind_r_1_2 = self.ind_gated_embedding_fusion(ind_r_1_2)

        # Ecological hierarchy
        z_r_1_2 = torch.cat([row_1, row_2], dim=1)
        z_r_1_2 = self.eco_cnn(z_r_1_2).view(batch_size, -1)
        eco_r_1_2 = torch.cat([ind_r_1_2, z_r_1_2], dim=1)
        eco_r_1_2 = self.eco_gated_embedding_fusion(eco_r_1_2)
        dominant_embedding = eco_r_1_2

        embeddings = torch.zeros(batch_size, 8, 2 * self.eco_output_size, device=x.device).type_as(x)
        for i in range(8):
            # Cell-wise hierarchy
            cell_r_3 = x_cell[:, [6, 7, 8 + i], :].reshape(batch_size, -1)
            cell_r_3 = self.cell_gated_embedding_fusion(cell_r_3)

            # Individual-wise hierarchy
            row_3 = x[:, [6, 7, 8 + i], :, :]
            ind_r_3 = self.ind_cnn(row_3).view(batch_size, -1)

            ind_r_1_3 = torch.cat([cell_r_1, ind_r_1, cell_r_3, ind_r_3], dim=1)
            ind_r_1_3 = self.ind_gated_embedding_fusion(ind_r_1_3)

            ind_r_2_3 = torch.cat([cell_r_2, ind_r_2, cell_r_3, ind_r_3], dim=1)
            ind_r_2_3 = self.ind_gated_embedding_fusion(ind_r_2_3)

            # Ecological hierarchy
            z_r_1_3 = torch.cat([row_1, row_3], dim=1)
            z_r_1_3 = self.eco_cnn(z_r_1_3).view(batch_size, -1)
            eco_r_1_3 = torch.cat([ind_r_1_3, z_r_1_3], dim=1)
            eco_r_1_3 = self.eco_gated_embedding_fusion(eco_r_1_3)

            z_r_2_3 = torch.cat([row_2, row_3], dim=1)
            z_r_2_3 = self.eco_cnn(z_r_2_3).view(batch_size, -1)
            eco_r_2_3 = torch.cat([ind_r_2_3, z_r_2_3], dim=1)
            eco_r_2_3 = self.eco_gated_embedding_fusion(eco_r_2_3)

            candidate_embedding = 0.5 * (eco_r_1_3 + eco_r_2_3)
            candidate_embedding = torch.cat([dominant_embedding, candidate_embedding], dim=1)
            embeddings[:, i, :] = self.embedding(candidate_embedding)

        return embeddings
