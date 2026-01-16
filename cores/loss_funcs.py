import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_scatter import scatter

EPS = 1e-6


class CentralNodeEmbedPrediction(nn.Module):
    def __init__(self, reduction: str = "mean"):
        super(CentralNodeEmbedPrediction, self).__init__()
        assert reduction in ['mean', 'sum'], "reduction must be 'mean' or 'sum'"
        self.reduction = reduction

    def forward(self, z, frame, edge_index, batch_size: int = None):
        """

        :param z: [N, d]
        :param frame: [N, r, d]
        :param edge_index: [E, ]
        :param batch_size:
        :return: loss
        """
        x = torch.einsum('ikj, ij -> ik', frame, z)  # [N, r]
        x_norm = F.normalize(x, dim=-1, p=2)

        src, dst = edge_index[0], edge_index[1]
        x_norm_neighbor = x_norm[src]

        x_pred = scatter(x_norm_neighbor, dst, dim=0, dim_size=x.shape[0], reduce=self.reduction)

        if batch_size is not None:
            x_norm = x_norm[: batch_size]
            x_pred = x_pred[: batch_size]
            frame = frame[: batch_size]

        x_error = x_norm - x_pred
        z_error = torch.einsum('ikj, ik -> ij', frame, x_error)  # [N, d]

        return (z_error ** 2).sum(-1).mean()