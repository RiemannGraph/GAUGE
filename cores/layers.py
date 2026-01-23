import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_mean, scatter_sum

EPS = 1e-6


class FeedForwardLayer(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, bias, act_str='gelu', drop=0.3):
        super().__init__()
        self.norm = nn.LayerNorm(in_dim)  # ← 新增
        self.layer = nn.Sequential(
            nn.Linear(in_dim, hid_dim, bias=bias),
            ActivateModule(act_str),
            nn.Dropout(drop),
            nn.Linear(hid_dim, out_dim, bias=bias),
            nn.Dropout(drop)
        )

    def forward(self, x):
        x = self.norm(x)
        x = self.layer(x)
        return x


class MultiPathTrivialization(nn.Module):
    def __init__(self, in_dim: int, fiber_dim, out_dim: int,
                 bias: bool = True, norm_str: str = "ln",
                 act_str: str = "gelu", drop=0.1, temperature=1.0):
        super().__init__()
        self.fiber_dim = fiber_dim
        self.dropout = nn.Dropout(drop)
        self.temperature = temperature
        self.path_lin = nn.Linear(in_dim, out_dim * fiber_dim, bias=bias)
        self.score_lin = nn.Sequential(
            nn.Linear(2 * in_dim, fiber_dim, bias=bias),
            nn.LeakyReLU()
        )
        self.norm = NormModule(norm_str, out_dim)
        self.fc = FeedForwardLayer(out_dim, out_dim, out_dim, bias, act_str, drop)

    def forward(self, x, edge_index):
        N = x.shape[0]
        src, dst = edge_index[0], edge_index[1]

        x_multi_path = self.path_lin(x).reshape(N, self.fiber_dim, -1)  # [N, r, d]
        s_ij = self.score_lin(torch.cat([x[src], x[dst]], dim=-1))
        s_ij = torch.softmax(s_ij / self.temperature, dim=-1).unsqueeze(-1)  # [E, r, 1]
        div = scatter_sum(s_ij, index=dst, dim_size=N, dim=0)  # [N, r]

        x_agg = scatter_sum(s_ij * x_multi_path[src], index=dst, dim_size=N, dim=0) / div.clamp(EPS)  # [N, r, d]

        x = x.unsqueeze(1) - x_agg
        x = self.fc(x)  # [N, r, d]
        x = self.norm(x)
        trilvial = torch.linalg.qr(x.transpose(-1, -2))[0].transpose(-1, -2)
        return trilvial


class GatedEnergyFlatten(nn.Module):
    def __init__(self, gamma=0.01):
        super().__init__()
        self.gamma = gamma
        self.beta = nn.Parameter(torch.tensor([0.0]))
        self.bias = nn.Parameter(torch.tensor([1.0]))

    def forward(self, trivial, edge_index):
        """

        params trivial: [N, r, d]
        """
        N, r, d = trivial.shape
        src, dst = edge_index[0], edge_index[1]
        tri_src, tri_dst = trivial[src], trivial[dst]
        tr_ij = r - (tri_src * tri_dst).sum(-1).sum(-1)  # [E, ]
        g_ij = torch.sigmoid(-F.softplus(self.beta) * tr_ij + self.bias).unsqueeze(-1).unsqueeze(-1)

        tri_tmp = scatter_mean(g_ij * tri_src, index=dst, dim=0, dim_size=N) * self.gamma  # [N, r, d]

        trivial = (1 - self.gamma) * trivial + tri_tmp

        trivial = torch.linalg.qr(trivial.transpose(-1, -2))[0].transpose(-1, -2)
        return trivial


class ActivateModule(nn.Module):
    ACTIVATION_MAP = {
        "relu": nn.ReLU,
        "tanh": nn.Tanh,
        "sigmoid": nn.Sigmoid,
        "elu": nn.ELU,
        "gelu": nn.GELU,
        "none": nn.Identity,
    }

    def __init__(self, act_str: str):
        super().__init__()
        self.act = self.ACTIVATION_MAP[act_str]()

    def forward(self, x):
        return self.act(x)


class NormModule(nn.Module):
    NORM_MAP = {
        "layer_norm": nn.LayerNorm,
        "batch_norm": nn.BatchNorm1d,
        "none": nn.Identity,
    }

    def __init__(self, norm_str: str, dim: int):
        super().__init__()
        self.norm = self.NORM_MAP[norm_str](dim)

    def forward(self, x):
        return self.norm(x)