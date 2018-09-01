import torch


class GroupNorm(torch.nn.Module):

    def __init__(self, n_features, n_groups=2):
        super().__init__()
        self.norm = torch.nn.GroupNorm(n_groups, n_features)

    def forward(self, x):
        return self.norm(x)

