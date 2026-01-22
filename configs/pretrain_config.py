import sys
import argparse
from dataclasses import dataclass
from typing import List
from configs.base_config import ModelConfig, add_model_config, load_config_from_yaml, save_config_to_yaml


@dataclass
class PretrainConfig(ModelConfig):
    # Training
    drop: float = 0.1
    batch_size: int = 128
    pretrain_epochs: int = 100
    lr_pretrain: float = 1e-3
    pretrain_weight_decay: float = 1e-5
    max_grad_norm: float = 1.0
    log_interval: int = 10
    save_interval: int = 10
    resume_checkpoint: bool = False
    resume_temp_checkpoint: bool = False
    warmup_epochs: int = 1

    # Path
    checkpoint_dir: str = "checkpoints/pretrain/"
    log_path: str = "logs/pretrain/pretrain.log"


def get_pretrain_parser():
    parser = argparse.ArgumentParser(description="Graph Pretraining Configuration")

    # Training
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size for data loading')
    parser.add_argument('--pretrain_epochs', type=int, default=20,
                        help='Total pretrain epochs')
    parser.add_argument('--drop', type=float, default=0.1,
                       help='Dropout rate')
    parser.add_argument('--lr_pretrain', type=float, default=3e-4,
                        help='Learning rate for pretraining')
    parser.add_argument('--pretrain_weight_decay', type=float, default=0,
                        help='Weight decay for Adam optimizer')
    parser.add_argument('--max_grad_norm', type=float, default=1.0,
                        help='Max gradient norm for clipping')
    parser.add_argument('--log_interval', type=int, default=100,
                        help='Log every N batches')
    parser.add_argument('--save_interval', type=int, default=1,
                        help='Save checkpoint every N epochs')
    parser.add_argument('--resume_checkpoint', action='store_true',
                        help='Whether to resume from latest checkpoint')
    parser.add_argument('--warmup_epochs', type=int, default=1,
                        help="Number of warmup epochs to compute prototype loss")

    # Config IO
    parser.add_argument('--save_config', action="store_true",
                        help='Whether to save current config as YAML (optional)')
    parser.add_argument('--config_load_path', type=str, default=None,
                        help='Path to load config from YAML (optional, will override cmd args)')

    add_model_config(parser)
    return parser


def parse_pretrain_config(remaining_argv=None) -> PretrainConfig:
    parser = get_pretrain_parser()
    args = parser.parse_args(remaining_argv)

    # If using YAML file
    if args.config_load_path:
        print(f"Loading config from YAML: {args.config_load_path}")
        yaml_config = load_config_from_yaml(args.config_load_path)

        for key, value in yaml_config.items():
            if hasattr(args, key):
                if not any(opt_str in str(sys.argv) for opt_str in [f'--{key}', f'-{key}']):
                    setattr(args, key, value)

    config = PretrainConfig(
        num_neighbors=args.num_neighbors,
        root=args.root,
        pretrain_single_graph_data=args.pretrain_single_graph_data,

        num_workers=args.num_workers,
        n_layers=args.n_layers,
        n_flat_layers=args.n_flat_layers,
        in_dim=args.in_dim,
        hid_dim=args.hid_dim,
        att_dim=args.att_dim,
        fiber_dim=args.fiber_dim,
        bias=args.bias,
        act_str=args.act_str,
        drop=args.drop,
        normalize=args.normalize,
        norm_str=args.norm_str,
        temperature=args.temperature,
        gamma=args.gamma,

        batch_size=args.batch_size,
        pretrain_epochs=args.pretrain_epochs,
        lr_pretrain=args.lr_pretrain,
        pretrain_weight_decay=args.pretrain_weight_decay,
        max_grad_norm=args.max_grad_norm,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        warmup_epochs=args.warmup_epochs,
        loss_reduction=args.loss_reduction
    )

    # Path
    config.log_path = "logs/pretrain/pretrain.log"
    dir_name = ""
    for d in config.pretrain_single_graph_data:
        dir_name += f"{d}_"

    config.checkpoint_dir = f"checkpoints/pretrain/{dir_name[:-1]}"
    config.log_path = f"logs/pretrain/{dir_name[:-1]}.log"

    if args.save_config:
        config_save_path = f"./scripts/pretrain/{dir_name[:-1]}.yaml"
        save_config_to_yaml(config, config_save_path)
    return config