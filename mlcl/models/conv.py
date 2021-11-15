from torch import nn


class ConvBnRelu(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super(ConvBnRelu, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return self.relu(x)
