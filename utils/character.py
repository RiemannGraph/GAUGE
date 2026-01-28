import numpy as np
import torch
import torch.nn.functional as F
from torch_scatter import scatter
import networkx as nx
from torch_geometric.utils import to_networkx, subgraph, from_networkx
from torch_geometric.data import Data, Batch


def predict_error(z, trivial, edge_index, batch_size: int = None, reduction: str = 'mean'):
    """

    :param z: [N, d]
    :param trivial: [N, r, d]
    :param edge_index: [E, ]
    :param batch_size:
    :param reduction:
    :return: loss
    """
    x_pred = torch.einsum('ikj, ij -> ik', trivial, z)
    x_target = torch.einsum('ikj, ij -> ik', trivial, z.detach())
    x_pred = F.normalize(x_pred, dim=-1, p=2)
    x_target = F.normalize(x_target, dim=-1, p=2)

    src, dst = edge_index[0], edge_index[1]

    x_pred = scatter(x_pred[src], dst, dim=0, dim_size=z.shape[0], reduce=reduction)  # [N, ]

    if batch_size is not None:
        x_pred = x_pred[: batch_size]
        x_target = x_target[: batch_size]

    loss = torch.sum((x_target - x_pred) ** 2, dim=-1)
    return loss


def get_character_structs(
        z,
        trivial,
        graph,
        K=10,
        min_component_size=3
):
    """
    Extract characteristic structures using error ranking.

    Returns:
        valid_components: List of node sets (each is a Tk)
        chi_Tk_list: List of mean errors for each Tk
        chi_G: Total characteristic number = sum(chi_Tk)
    """
    N  = z.shape[0]
    edge_index = graph.edge_index
    batch = graph.batch
    errors = predict_error(z, trivial, edge_index)

    loss_np = errors.detach().cpu().numpy()
    batch_np = batch.cpu().numpy()

    num_graphs = batch.max().item() + 1
    valid_structures = []

    data = Data(edge_index=edge_index)
    network_graph = to_networkx(data, to_undirected=True)

    for g_id in range(num_graphs):
        node_mask = (batch_np == g_id)
        node_indices_global = np.where(node_mask)[0]

        if len(node_indices_global) == 0:
            continue
        graph_loss = loss_np[node_mask]  # (n_g,)

        topk_indices = np.argsort(graph_loss)[:K]
        reliable_mask_local = np.zeros_like(graph_loss, dtype=bool)
        reliable_mask_local[topk_indices] = True

        reliable_nodes_global = node_indices_global[reliable_mask_local]

        if len(reliable_nodes_global) == 0:
            continue

        subgraph_g = network_graph.subgraph(node_indices_global)
        reliable_subgraph = subgraph_g.subgraph(reliable_nodes_global)

        components = list(nx.connected_components(reliable_subgraph))

        valid_comp = [comp for comp in components if len(comp) >= min_component_size]
        valid_structures.extend(valid_comp)

    chi_Tk_list = []
    comp_batches = []
    sub_graphs = []

    for comp in valid_structures:
        comp = list(comp)
        comp_errors = errors[comp]
        comp_batch = batch[comp][0: 1]
        chi_Tk = comp_errors.mean()

        chi_Tk_list.append(chi_Tk)
        comp_batches.append(comp_batch)
        sub_edge_index, _ = subgraph(comp, edge_index, num_nodes=N, relabel_nodes=True)
        sub_graphs.append(
            Data(edge_index=sub_edge_index, x=graph.x[comp])
        )

    chi_Tk = torch.stack(chi_Tk_list, dim=-1)
    comp_batches = torch.cat(comp_batches, dim=-1)
    T = Batch.from_data_list(sub_graphs)
    return chi_Tk, comp_batches, T