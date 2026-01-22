import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from cores.models import Characteron
from cores.loss_funcs import CharacteristicStructureLoss


class CharacteronAdapter(nn.Module):
    def __init__(self, configs,
                 feature_dim,
                 pretrained_model: Characteron,
                 task_type: str,
                 num_cls: int):
        """

        :param configs: PretrainConfig
        :param feature_dim:
        :param pretrained_model:
        :param task_type: [node_cls, graph_cls, edge_cls]
        :param num_cls: classes number
        """
        super(CharacteronAdapter, self).__init__()
        assert task_type in ["node_cls", "graph_cls", "link_cls"], "the task type must be one of [node_cls, graph_cls, link_cls]"
        self.configs = configs
        self.input_lin =nn.Sequential(
            nn.Linear(feature_dim, configs.in_dim),
            nn.LayerNorm(configs.in_dim)
            )
        self.pretrained_model = pretrained_model
        self.pretrained_model.frozen()
        self.head = ADAPTERS[task_type](configs.hid_dim, num_cls, configs.drop)
        # self.loss_fn = CharacteristicStructureLoss(configs.loss_reduction)

    def forward(self, graph: Data):
        graph.x = self.input_lin(graph.x)
        z, trivial, target = self.pretrained_model(graph, return_target=True)

        pred = self.head(z, trivial, graph)
        # loss = self.loss_fn(target, z, trivial, graph.edge_index, graph.batch_size if hasattr(graph, "batch_size") else None)
        return pred, 0.


class GraphClassificationHead(nn.Module):
    def __init__(self, hid_dim: int, num_classes: int, drop: float = 0.2):
        super(GraphClassificationHead, self).__init__()
        self.head = nn.Linear(hid_dim, num_classes)
        self.drop = nn.Dropout(drop)

    def forward(self, z: torch.Tensor, trivial, graph: Data):
        z = self.drop(z)
        return self.head(z)


class LinkClassificationHead(nn.Module):
    """
    For knowledge graph link prediction (edge classification / triple scoring)
    Using dot product or bilinear scoring.
    """
    def __init__(self, hid_dim: int, num_classes: int, drop: float = 0.2):
        super(LinkClassificationHead, self).__init__()
        self.drop = nn.Dropout(drop)

    def forward(self, z: torch.Tensor, trivial: torch.Tensor, graph: Data):
        z = F.normalize(z, p=2, dim=-1)
        edge_label_index = graph.edge_label_index
        src, dst = edge_label_index[0], edge_label_index[1]
        # z_src, x_dst = z[src], (trivial @ z.unsqueeze(-1))[dst].squeeze()
        # score = ((trivial[dst] @ z_src.unsqueeze(-1)).squeeze() * x_dst).sum(-1)
        score = (z[src] * z[dst]).sum(-1)
        return score


ADAPTERS = {
    'node_cls': GraphClassificationHead,
    'graph_cls': GraphClassificationHead,
    'link_cls': LinkClassificationHead,
}