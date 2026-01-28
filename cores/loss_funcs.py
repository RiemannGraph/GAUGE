import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_scatter import scatter, scatter_softmax, scatter_sum

EPS = 1e-6


class DirichletLoss(nn.Module):
    def __init__(self, reduction: str = "mean", temperature=1.0):
        super().__init__()
        assert reduction in ['mean', 'sum'], "reduction must be 'mean' or 'sum'"
        self.reduction = reduction
        self.temperature = temperature

    def forward(self, target, z, trivial, edge_index, batch_size: int = None):
        """

        :param target: [N, d]
        :param z: [N, d]
        :param trivial: [N, r, d]
        :param edge_index: [E, ]
        :param batch_size: PyG Data.batch_size
        :return: loss
        """
        x_pred = torch.einsum('ikj, ij -> ik', trivial, z)
        x_target = torch.einsum('ikj, ij -> ik', trivial, target.detach())
        x_pred = F.normalize(x_pred, dim=-1, p=2)
        x_target = F.normalize(x_target, dim=-1, p=2)

        src, dst = edge_index[0], edge_index[1]

        x_pred = scatter(x_pred[src], dst, dim=0, dim_size=z.shape[0], reduce=self.reduction)  # [N, ]

        if batch_size is not None:
            x_pred = x_pred[: batch_size]
            x_target = x_target[: batch_size]

        loss = torch.sum((x_target - x_pred) ** 2, dim=-1).mean()
        return loss