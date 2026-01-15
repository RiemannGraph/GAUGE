import torch
import torch.nn as nn
import torch.nn.functional as F

from cores.models import GraphTrivializeModel
from torch_geometric.data import Data


class GraphTrivialAdapter(nn.Module):
    def __init__(self, configs,
                 feature_dim,
                 pretrained_model: GraphTrivializeModel,
                 task_type: str,
                 num_cls: int):
        """

        :param configs: PretrainConfig
        :param feature_dim:
        :param pretrained_model:
        :param task_type: [node_cls, graph_cls, edge_cls]
        :param num_cls: classes number
        """
        super(GraphTrivialAdapter, self).__init__()
        assert task_type in ["node_cls", "graph_cls", "link_cls"], "the task type must be one of [node_cls, graph_cls, link_cls]"
        self.configs = configs
        self.input_lin = nn.Linear(feature_dim, configs.in_dim)
        self.pretrained_model = pretrained_model
        self.pretrained_model.frozen()
        self.prompt_z = nn.Parameter(torch.empty(configs.hid_dim, configs.hid_dim))
        nn.init.orthogonal_(self.prompt_z.data)

        self.head = ADAPTERS[task_type](configs.hid_dim, num_cls, configs.drop)

    def forward(self, graph: Data):
        graph.x = self.input_lin(graph.x)
        z, frame = self.pretrained_model(graph)

        z_adapt = z @ self.prompt_z

        pred = self.head(z_adapt, graph)
        return pred


class GraphClassificationHead(nn.Module):
    def __init__(self, hid_dim: int, num_classes: int, drop: float = 0.2):
        super(GraphClassificationHead, self).__init__()
        self.head = nn.Linear(hid_dim, num_classes)
        self.drop = nn.Dropout(drop)

    def forward(self, z: torch.Tensor, graph: Data):
        z = self.drop(z)
        return self.head(z)


class LinkClassificationHead(nn.Module):
    """
    For knowledge graph link prediction (edge classification / triple scoring)
    Using dot product or bilinear scoring.
    """
    def __init__(self, hid_dim: int, num_classes: int, drop: float = 0.2):
        super(LinkClassificationHead, self).__init__()
        self.score_fn = nn.Bilinear(hid_dim, hid_dim, num_classes)
        self.drop = nn.Dropout(drop)

    def forward(self, z: torch.Tensor, graph: Data):
        z = self.drop(z)
        src_emb = z[::2]
        dst_emb = z[1::2]
        src_emb = F.normalize(src_emb, p=2, dim=1)
        dst_emb = F.normalize(dst_emb, p=2, dim=1)
        return self.score_fn(src_emb, dst_emb)


ADAPTERS = {
    'node_cls': GraphClassificationHead,
    'graph_cls': GraphClassificationHead,
    'link_cls': LinkClassificationHead,
}