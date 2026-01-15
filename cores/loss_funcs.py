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
        x = torch.einsum('ikj, ij -> ik', frame, z) # [N, r]
        x_norm = F.normalize(x, dim=-1, p=2)
        z_norm = torch.einsum('ikj, ik -> ij', frame, x_norm)   # [N, d]

        src, dst = edge_index[0], edge_index[1]
        z_norm_neighbor = z_norm[src]

        z_pred = scatter(z_norm_neighbor, dst, dim=0, dim_size=z.shape[0], reduce=self.reduction)

        if batch_size is not None:
            return F.mse_loss(z_pred[: batch_size], z_norm[: batch_size])
        else:
            return F.mse_loss(z_pred, z_norm)