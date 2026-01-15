import torch
import torch.nn as nn
from torch_geometric.data import Data
from cores.layers import NormModule, FeedForwardLayer, MultiHeadMPLayer, FrameSmoothModule
from cores.loss_funcs import CentralNodeEmbedPrediction


EPS = 1e-6


class GraphTrivializeLayer(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.multi_head_mp = MultiHeadMPLayer(configs.fiber_dim, configs.hid_dim,
                                              configs.hid_dim, configs.bias,
                                              configs.norm_str, configs.act_str, configs.drop)
        self.f_smooth = FrameSmoothModule(configs.n_smooth_layers, configs.fiber_dim,
                                          configs.hid_dim, configs.bias,
                                          configs.norm_str, configs.act_str, configs.drop)
        self.horizon_lin = nn.Linear(configs.hid_dim, configs.hid_dim, bias=False)
        self.vertical_lin = nn.Linear(configs.hid_dim, configs.hid_dim, bias=configs.bias)

    def forward(self, z, edge_index, return_frame: bool = False):
        z, frame = self.multi_head_mp(z, edge_index)    # [N, d]
        frame = self.f_smooth(frame)    # [N, r, d]
        x = self.horizon_lin(torch.einsum('ikj, ij->ik', frame, z)) # [N, r]
        z = torch.einsum('ikj, ik->ij', frame, x) + self.vertical_lin(z)
        if return_frame:
            return z, frame
        else:
            return z


class GraphTrivializeModel(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.input_lin = nn.Linear(configs.in_dim, configs.hid_dim)
        self.layers = nn.ModuleList([GraphTrivializeLayer(configs) for _ in range(configs.n_layers)])
        self.loss_fn = CentralNodeEmbedPrediction(configs.loss_reduction)

    def forward(self, graph: Data):
        z, edge_index = graph.x, graph.edge_index
        z = self.input_lin(z)
        for layer in self.layers[:-1]:
            z = layer(z, edge_index)
        z, frame = self.layers[-1](z, edge_index, return_frame=True)
        return z, frame

    def loss(self, z, frame, graph):
        return self.loss_fn(z, frame, graph.edge_index)