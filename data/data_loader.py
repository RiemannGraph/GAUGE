import torch_geometric.transforms as T
from torch_geometric.datasets import (
    Reddit, AttributedGraphDataset,
    Planetoid, Amazon, FacebookPagePage
)
from ogb.nodeproppred import PygNodePropPredDataset
from data.data_transform import FlattenLabels, UnifyFeatureDims
from data.data_process import graph_few_shot_splits


def load_pretrain_single_graph_data(configs, data_name: str):
    root = configs.root
    transform = T.Compose([
        T.ToUndirected(),
        UnifyFeatureDims(configs.in_dim)
    ])
    if data_name == "ogbn-arxiv":
        dataset = PygNodePropPredDataset(root=root, name=data_name, transform=transform)
    elif data_name in ["Computers", "Photo"]:
        dataset = Amazon(root, data_name, transform=transform)
    elif data_name == 'Reddit':
        dataset = Reddit(f"{root}/{data_name}", transform=transform)
    elif data_name == "FacebookPagePage":
        dataset = FacebookPagePage(f"{root}/{data_name}", transform=transform)
    elif data_name == 'PPI':
        dataset = AttributedGraphDataset(root, name=data_name.lower(), transform=transform)
    elif data_name in ["Cora", "CiteSeers", "PubMed"]:
        dataset = Planetoid(root, data_name, transform=transform)
    else:
        raise ValueError('Invalid data_name')
    data = dataset[0]
    return data


def load_few_shot_single_graph_data(configs, data_name, k_shot, num_splits, num_val=0.1):
    root = configs.root
    transform = T.Compose([
        FlattenLabels(),
        T.RandomNodeSplit(split='test_rest', num_splits=num_splits,
                          num_train_per_class=k_shot, num_val=num_val)
    ])
    if data_name == "ogbn-arxiv":
        dataset = PygNodePropPredDataset(root=root, name=data_name, transform=T.Compose([T.ToUndirected(), transform]))
    elif data_name in ["Cora", "CiteSeers", "PubMed"]:
        dataset = Planetoid(root, data_name, transform=transform)
    elif data_name in ["Computers", "Photo"]:
        dataset = Amazon(root, data_name, transform=transform)
    elif data_name == 'Reddit':
        dataset = Reddit(f"{root}/{data_name}", transform=transform)
    elif data_name == "FacebookPagePage":
        dataset = FacebookPagePage(f"{root}/{data_name}", transform=transform)
    elif data_name == 'PPI':
        dataset = AttributedGraphDataset(root, name=data_name.lower(), transform=transform)
    else:
        raise ValueError('Invalid data_name')
    data = dataset[0]
    train_mask, val_mask, test_mask = data.train_mask, data.val_mask, data.test_mask
    return dataset, data, train_mask, val_mask, test_mask