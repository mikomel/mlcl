from torch import nn


class LinearBnRelu(nn.Module):

    def __init__(self, in_dim, out_dim):
        super(LinearBnRelu, self).__init__()
        self.out_dim = out_dim
        self.linear = nn.Linear(in_dim, out_dim)
        self.bn = nn.BatchNorm1d(out_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        if x.ndim == 2:
            x = self.linear(x)
            x = self.bn(x)
        elif x.ndim == 3:
            batch_size, num_objects, object_size = x.size()
            x = x.view(batch_size * num_objects, object_size)
            x = self.linear(x)
            x = self.bn(x)
            x = x.view(batch_size, num_objects, self.out_dim)
        elif x.ndim == 4:
            batch_size, num_panels, num_objects, object_size = x.size()
            x = x.reshape(batch_size * num_panels * num_objects, object_size)
            x = self.linear(x)
            x = self.bn(x)
            x = x.reshape(batch_size, num_panels, num_objects, self.out_dim)
        else:
            raise Exception()
        return self.relu(x)


class DeepLinearBnRelu(nn.Module):

    def __init__(self, depth, in_dim, out_dim, change_dim_first=True):
        super(DeepLinearBnRelu, self).__init__()
        assert depth >= 2, 'use LinearBnRelu for 1-layer block'
        layers = []
        if change_dim_first:
            layers += [LinearBnRelu(in_dim, out_dim)]
            for _ in range(depth - 1):
                layers += [LinearBnRelu(out_dim, out_dim)]
        else:
            for _ in range(depth - 1):
                layers += [LinearBnRelu(in_dim, in_dim)]
            layers += [LinearBnRelu(in_dim, out_dim)]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
