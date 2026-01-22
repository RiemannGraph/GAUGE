import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_scatter import scatter

EPS = 1e-6


class CharacteristicStructureLoss(nn.Module):
    def __init__(self, reduction: str = "mean"):
        super().__init__()
        assert reduction in ['mean', 'sum'], "reduction must be 'mean' or 'sum'"
        self.reduction = reduction

    def forward(self, target, z, trivial, edge_index, batch_size: int = None):
        """

        :param target: [N, d]
        :param z: [N, d]
        :param trivial: [N, r, d]
        :param edge_index: [E, ]
        :param batch_size: PyG Data.batch_size
        :return: loss
        """
        Qtz = torch.einsum('ikj, ij -> ik', trivial, z)
        z = torch.einsum('ikj, ik -> ij', trivial, Qtz)
        z_norm = F.normalize(z, dim=-1, p=2)
        target = F.normalize(target, dim=-1, p=2).detach()

        src, dst = edge_index[0], edge_index[1]
        z_norm_neighbor = z_norm[src]

        z_pred = scatter(z_norm_neighbor, dst, dim=0, dim_size=z.shape[0], reduce=self.reduction)

        if batch_size is not None:
            target = target[: batch_size]
            z_pred = z_pred[: batch_size]

        loss = torch.sum((target - z_pred) ** 2, dim=-1).mean()

        return loss