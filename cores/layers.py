import torch
import torch.nn as nn
from torch_scatter import scatter_mean
from torch_geometric.nn import GCNConv


class FeedForwardLayer(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, bias, act_str='gelu', drop=0.3):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(in_dim, hid_dim, bias=bias),
            nn.Dropout(drop),
            ActivateModule(act_str),
            nn.Linear(hid_dim, out_dim, bias=bias),
            nn.Dropout(drop)
        )

    def forward(self, x):
        x = self.layer(x)
        return x


class MultiHeadMPLayer(nn.Module):
    def __init__(self, fiber_dim, in_dim: int, out_dim: int,
                 bias: bool = True, norm_str: str = "ln",
                 act_str: str = "relu", drop=0.1):
        super().__init__()
        self.out_dim = out_dim
        self.fiber_dim = fiber_dim
        self.dropout = nn.Dropout(drop)
        self.head_lin = nn.Linear(in_dim, out_dim * fiber_dim, bias=bias)
        self.score_lin = nn.Linear(2 * out_dim, 1, bias=False)
        self.fc = FeedForwardLayer(out_dim, out_dim, out_dim, bias, act_str, drop)

    def forward(self, x, edge_index):
        x_multi_head = self.dropout(self.head_lin(x)).reshape(-1, self.fiber_dim, self.out_dim)    # [N, r, d]
        src, dst = edge_index[0], edge_index[1]
        x_src, x_dst = x_multi_head[src], x_multi_head[dst]  # [E, r, d]
        scores = self.score_lin(torch.cat([x_src, x_dst], dim=-1)).softmax(1)  # [E, r, 1]
        x = scatter_mean(scores * x_src, index=dst, dim=0)  # [N, r, d]
        x = self.fc(x)  # [N, r, d]
        frame = torch.qr(x.transpose(-1, -2))[0].transpose(-1, -2)
        return x, frame


class FrameSmoothModule(nn.Module):
    def __init__(self, n_layers: int, fiber_dim: int, hid_dim: int,
                 bias: bool = True, norm_str: str = "ln",
                 act_str: str = "relu", drop=0.1):
        super().__init__()
        self.fiber_dim = fiber_dim
        self.squeeze_lin = nn.Linear(fiber_dim * hid_dim, hid_dim, bias=bias)
        self.layers = nn.ModuleList([
            FrameSmoothLayer(hid_dim, bias=bias, norm_str=norm_str, drop=drop)
        ])
        for _ in range(n_layers - 1):
            self.layers.append(
                FrameSmoothLayer(hid_dim, bias=bias, norm_str=norm_str, drop=drop)
            )

        self.out_fc = FeedForwardLayer(hid_dim, hid_dim, hid_dim, bias, act_str, drop)
        self.out_norm = NormModule(norm_str, hid_dim)
        self.lift_lin = nn.Linear(hid_dim, fiber_dim * hid_dim, bias=bias)

    def forward(self, frame, edge_index):
        f_vec = frame.reshape(frame.shape[0], -1)
        f_vec = self.squeeze_lin(f_vec)
        for layer in self.layers:
            f_vec = layer(f_vec, edge_index)
        f_vec = self.out_fc(f_vec)
        f_vec = self.out_norm(f_vec)
        f_vec = self.lift_lin(f_vec)

        f = f_vec.reshape(f_vec.shape[0], self.fiber_dim, -1)
        f = torch.qr(f.transpose(-1, -2))[0].transpose(-1, -2)
        return f


class FrameSmoothLayer(nn.Module):
    def __init__(self, hid_dim, bias: bool = True,
                 norm_str: str = "ln", drop=0.1):
        super().__init__()
        self.dropout = nn.Dropout(drop)
        self.gated_lin = nn.Linear(hid_dim, hid_dim, bias=False)
        self.agg = GCNConv(hid_dim, hid_dim, bias=bias, normalize=True)
        self.out_norm = NormModule(norm_str, hid_dim)

    def forward(self, f_vec, edge_index):
        """

        params f_vec: [N, r * d]
        """
        gates = self.gated_score(f_vec, edge_index)
        f_vec = gates * self.agg(f_vec, edge_index) + (1 - gates) * self.dropout(f_vec)
        f_vec = self.out_norm(f_vec)
        return f_vec

    def gated_score(self, f, edge_index):
        f = self.gated_lin(f)
        src, dst = edge_index[0], edge_index[1]
        f = f - scatter_mean(f[src], dst, dim=0)
        return torch.sigmoid(f)


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