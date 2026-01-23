import numpy as np
import torch
import torch.nn.functional as F
from torch_scatter import scatter
import networkx as nx
from torch_geometric.utils import to_networkx, subgraph, from_networkx
from torch_geometric.data import Data


def predict_error(z, trivial, edge_index, batch_size: int = None, reduction: str = 'mean'):
    """

    :param z: [N, d]
    :param trivial: [N, r, d]
    :param edge_index: [E, ]
    :param batch_size:
    :param reduction:
    :return: loss
    """
    Qtz = torch.einsum('ikj, ij -> ik', trivial, z)
    z = torch.einsum('ikj, ik -> ij', trivial, Qtz)
    z_norm = F.normalize(z, dim=-1, p=2)
    target = z_norm.clone()

    src, dst = edge_index[0], edge_index[1]
    z_norm_neighbor = z_norm[src]

    z_pred = scatter(z_norm_neighbor, dst, dim=0, dim_size=z.shape[0], reduce=reduction)

    if batch_size is not None:
        target = target[: batch_size]
        z_pred = z_pred[: batch_size]

    loss = torch.sum((target - z_pred) ** 2, dim=-1)
    return loss


def get_characteristic_structures(
        z,
        trivial,
        edge_index,
        y=None,
        percentile=90,
        min_component_size=3
):
    """
    Extract characteristic structures using error ranking.

    Returns:
        valid_components: List of node sets (each is a Tk)
        chi_Tk_list: List of mean errors for each Tk
        chi_G: Total characteristic number = sum(chi_Tk)
    """
    errors = predict_error(z, trivial, edge_index)

    threshold = np.percentile(errors.detach().cpu().numpy(), percentile)
    reliable_mask = (errors <= threshold).cpu().numpy()

    data = Data(x=z.detach(), edge_index=edge_index, y=y)
    nx_graph = to_networkx(data, to_undirected=True)

    reliable_nodes = np.where(reliable_mask)[0].tolist()
    reliable_subgraph = nx_graph.subgraph(reliable_nodes)

    candidate_components = list(nx.connected_components(reliable_subgraph))
    valid_components = [comp for comp in candidate_components if len(comp) >= min_component_size]

    chi_Tk_list = []
    chi_G = 0
    for comp in valid_components:
        comp_errors = errors[list(comp)]
        chi_Tk = comp_errors.mean()
        chi_Tk_list.append(chi_Tk)
        chi_G += chi_Tk

    chi_G = chi_G / len(valid_components)  # scalar
    chi_Tk = torch.stack(chi_Tk_list, dim=-1)

    return valid_components, chi_Tk, chi_G