from dataclasses import dataclass
from typing import Optional, Dict, Any, List
from argparse import ArgumentParser
import yaml
import os


@dataclass
class ModelConfig:
    """Shared pretraining datasets"""
    pretrain_graph_data: List[str] = None
    root: str = "./datasets"
    num_neighbors: Optional[List[int]] = None

    """Shared model architecture configuration"""

    n_layers: int = 2
    n_flat_layers: int = 2
    in_dim: int = 128
    hid_dim: int = 256
    att_dim: int = 512
    fiber_dim: int = 32
    bias: bool = True
    act_str: str = "gelu"

    normalize: bool = True
    norm_str: str = "layer_norm"

    temperature: float = 1.0
    gamma: float = 0.01

    loss_reduction: str = 'mean'

    """Shared Loader"""
    num_workers: int = 2


def add_model_config(parser: ArgumentParser):
    """Add shared model architecture arguments"""
    group = parser.add_argument_group("Model Architecture")
    parser.add_argument("--root", type=str, default="./datasets", help="Root directory for datasets.")
    group.add_argument('--pretrain_graph_data', type=str, nargs='+',
                       default=["HIV", "PROTEINS"],
                       help='node-level pretraining datasets')
    parser.add_argument('--num_neighbors', type=int, nargs="+", default=[20, 10],
                        help='maximum number of nodes per graph')

    # model configurations
    group.add_argument('--n_layers', type=int, default=2,
                       help='Number of GFM layers')
    group.add_argument('--n_flat_layers', type=int, default=2,
                       help='Number of trivialization flatten layers')
    group.add_argument('--in_dim', type=int, default=128,
                       help='Input feature dimension')
    group.add_argument('--hid_dim', type=int, default=512,
                       help='Hidden dimension')
    group.add_argument('--att_dim', type=int, default=512,
                       help='Attention dimension (if used)')
    group.add_argument('--fiber_dim', type=int, default=16,
                       help='Number of generators in FM')
    group.add_argument('--act_str', type=str, default='gelu',
                       help='Activation function')
    group.add_argument('--normalize', action='store_true',
                       help='Whether to normalize adjacency matrix')
    group.add_argument('--gamma', type=float, default=0.01,
                       help="EMA momentum")
    group.add_argument('--bias', action='store_false',
                       help='Whether to add bias term')
    group.add_argument('--norm_str', type=str, default='layer_norm', choices=['layer_norm', 'batch_norm'],
                       help="Normalization type")
    group.add_argument('--temperature', type=float, default=1.0,
                       help='Temperature')
    parser.add_argument('--loss_reduction', type=str, default='mean',
                        help='Type of loss reduction')

    parser.add_argument('--num_workers', type=int, default=0,
                        help='Number of workers for data loading')
    return parser


def save_config_to_yaml(config: ModelConfig, filepath: str):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    exclude_keys = {'resume_checkpoint', 'resume_temp_checkpoint'}
    filtered_config = {k: v for k, v in config.__dict__.items() if k not in exclude_keys}
    with open(filepath, 'w', encoding='utf-8') as f:
        yaml.dump(filtered_config, f, default_flow_style=False, indent=2, sort_keys=False)
    print(f"Config saved to {filepath}")


def load_config_from_yaml(filepath: str) -> Dict[str, Any]:
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Config file not found: {filepath}")
    with open(filepath, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)