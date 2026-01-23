import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_scatter import scatter_mean
from cores.layers import MultiPathTrivialization, GatedEnergyFlatten
from cores.loss_funcs import CharacteristicStructureLoss

EPS = 1e-6


class CharacteronLayer(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.multi_path_layer = MultiPathTrivialization(configs.hid_dim, configs.fiber_dim,
                                                        configs.hid_dim, configs.bias,
                                                        configs.norm_str, configs.act_str,
                                                        configs.drop, configs.temperature)
        self.gated_energy_layers = nn.ModuleList([
            GatedEnergyFlatten(configs.gamma)
            for _ in range(configs.n_flat_layers)
        ])
        self.fc = nn.Linear(configs.hid_dim, configs.hid_dim, bias=False)

    def forward(self, z, edge_index):
        trivial = self.multi_path_layer(z, edge_index)
        for layer in self.gated_energy_layers:
            trivial = layer(trivial, edge_index)
        Qtz = torch.einsum('ikj, ij -> ik', trivial, z)
        z = torch.einsum('ikj, ik -> ij', trivial, Qtz)
        src, dst = edge_index[0], edge_index[1]
        z = self.fc(z)
        z = scatter_mean(z[src], dst, dim=0, dim_size=z.shape[0])
        return z, trivial


class Characteron(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.n_layers = configs.n_layers
        self.input_lin = nn.Sequential(
            nn.Linear(configs.in_dim, configs.hid_dim),
            nn.LayerNorm(configs.hid_dim)
        )
        self.layers = nn.ModuleList([CharacteronLayer(configs)
                                     for _ in range(configs.n_layers)])
        self.loss_fn = CharacteristicStructureLoss(configs.loss_reduction)

    def forward(self, graph: Data, encoder: nn.Module = None, return_target: bool = False):
        edge_index = graph.edge_index
        if encoder is not None:
            z = encoder(graph)
        else:
            z = graph.x
        z = self.input_lin(z)

        if return_target:
            z0 = z.clone()

        trivial = None
        for i, layer in enumerate(self.layers):
            z, trivial = layer(z, edge_index)

        if return_target:
            return z, trivial, z0
        else:
            return z, trivial

    def frozen(self):
        for param in self.parameters():
            param.requires_grad_(False)

    def unfrozen(self):
        for param in self.parameters():
            param.requires_grad_(True)