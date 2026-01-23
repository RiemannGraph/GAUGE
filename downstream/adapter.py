import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from cores.models import Characteron
from cores.loss_funcs import CharacteristicStructureLoss
from torch_geometric.nn import global_mean_pool


class Adapter(nn.Module):
    def __init__(self, configs,
                 feature_dim,
                 pretrained_model: Characteron,
                 num_cls: int):
        """

        :param configs: PretrainConfig
        :param feature_dim:
        :param pretrained_model:
        :param num_cls: classes number
        """
        super().__init__()
        self.configs = configs
        self.input_lin = nn.Sequential(
            nn.Linear(feature_dim, configs.in_dim),
            nn.LayerNorm(configs.in_dim)
        )
        self.pretrained_model = pretrained_model
        self.loss_fn = CharacteristicStructureLoss(configs.loss_reduction)

class NodeAdapter(Adapter):
    def __init__(self, configs,
                 feature_dim,
                 pretrained_model: Characteron,
                 num_cls: int):
        super().__init__(configs, feature_dim, pretrained_model, num_cls)
        self.head = nn.Linear(configs.hid_dim, num_cls)

    def forward(self, graph: Data):
        graph.x = self.input_lin(graph.x)
        z, trivial = self.pretrained_model(graph)

        pred = self.head(z)
        loss = self.loss_fn(z, z, trivial, graph.edge_index, graph.batch_size if hasattr(graph, "batch_size") else None)
        return pred, loss * 0.1


class GraphAdapter(Adapter):
    def __init__(self, configs,
                 feature_dim,
                 pretrained_model: Characteron,
                 num_cls: int):
        super().__init__(configs, feature_dim, pretrained_model, num_cls)
        self.input_lin = nn.Sequential(
            nn.Linear(feature_dim, configs.in_dim - feature_dim),
            nn.LayerNorm(configs.in_dim - feature_dim)
        )
        self.head = nn.Linear(configs.hid_dim, num_cls)

    def forward(self, graph: Data):
        pe = graph.x
        x = self.input_lin(graph.x)
        graph.x = torch.cat([x, pe], dim=-1)
        z, trivial, target = self.pretrained_model(graph, return_target=True)
        loss = self.loss_fn(target, z, trivial, graph.edge_index, None)
        z = global_mean_pool(z, graph.batch, size=len(graph))
        pred = self.head(z)
        return pred, loss * 0.1


class LinkAdapter(Adapter):
    def __init__(self, configs,
                 feature_dim,
                 pretrained_model: Characteron,
                 num_cls: int):
        super().__init__(configs, feature_dim, pretrained_model, num_cls)
        self.beta = nn.Parameter(torch.tensor([0.0]))
        self.bias = nn.Parameter(torch.tensor([1.0]))

    def forward(self, graph: Data):
        graph.x = self.input_lin(graph.x)
        z, trivial, target = self.pretrained_model(graph, return_target=True)

        z = F.normalize(z, p=2, dim=-1)
        edge_label_index = graph.edge_label_index
        src, dst = edge_label_index[0], edge_label_index[1]
        eye = torch.eye(trivial.shape[1]).unsqueeze(0).to(z.device)
        tr_ij = trivial[src] @ trivial[dst].transpose(-1, -2) - eye  # [E, ]
        pred = torch.frobenius_norm(tr_ij, dim=(-1, -2))
        return pred, 0