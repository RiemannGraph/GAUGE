import argparse
import sys
from dataclasses import dataclass
from typing import List
from configs.base_config import ModelConfig, load_config_from_yaml, save_config_to_yaml, add_model_config


@dataclass
class AdaptionConfig(ModelConfig):
    # Data
    data_name: str = "PubMed"
    pretrained_checkpoint: str = "checkpoints/pretrain/pretrain_final_model.pth"

    # Task
    task_type: str = "node_cls"
    task_types: List[str] = None
    k_shot: int = 5
    num_way_link: int = 10
    num_trials: int = 10
    num_val: float = 0.1
    metric: str = "acc"

    # Training
    drop: int = 0.1
    batch_size: int = 128
    lr_task: float = 1e-3
    task_weight_decay: float = 1e-5
    task_epochs: int = 500
    max_grad_norm: float = 1.0
    eval_interval: int = 10
    resume_checkpoint: bool = False
    patience: int = 20

    # Path
    checkpoint_dir: str = None
    log_path: str = None


def get_adaption_parser():
    parser = argparse.ArgumentParser(description="Graph Downstream Adaption Configuration")

    parser.add_argument("--data_name", type=str, default="PubMed",
                        help="Name of the dataset. [ogbn-arxiv, Computers, Reddit, FB15k_237, PROTEINS, HIV] ")
    parser.add_argument("--task_type", type=str, default="node_cls", choices=["node_cls", "graph_cls", "link_cls"],
                        help="Type of downstream task.")
    parser.add_argument("--pretrained_checkpoint", type=str,
                        default="checkpoints/pretrain/Computers_Amazon-ratings/pretrain_epoch_12.pth",
                        help="file path of pretrained model checkpoint.")
    parser.add_argument("--metric", type=str, default="acc", choices=["acc", "auc"])

    # Task
    parser.add_argument("--k_shot", type=int, default=5,
                        help="Number of shots in few-shot learning.")
    parser.add_argument("--num_way_link", type=int, default=10,
                        help="Number of ways in link classification few-shot learning.")
    parser.add_argument("--num_trials", type=int, default=5,
                        help="Number of independent trials.")
    parser.add_argument("--num_val", type=float, default=0.1,
                        help="Proportion of validation set.")

    # Training
    parser.add_argument("--drop", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=512,
                        help="Batch size for task training.")
    parser.add_argument("--lr_task", type=float, default=1e-3,
                        help="Learning rate for task model.")
    parser.add_argument("--task_weight_decay", type=float, default=0,
                        help="Weight decay for task optimizer.")
    parser.add_argument("--task_epochs", type=int, default=1000,
                        help="Number of epochs for task training.")
    parser.add_argument("--max_grad_norm", type=float, default=1.0,
                        help="Maximum gradient norm for clipping.")
    parser.add_argument("--eval_interval", type=int, default=10,
                        help="Log every N epochs.")
    parser.add_argument("--resume_checkpoint", action="store_true",
                        help="Whether to resume from checkpoint.")
    parser.add_argument("--patience", type=int, default=15,
                        help="Patience for early stopping.")

    # Config IO
    parser.add_argument('--save_config', action="store_false",
                        help='Whether to save current config as YAML (optional)')
    parser.add_argument('--config_load_path', type=str, default=None,
                        help='Path to load config from YAML (optional, will override cmd args)')

    add_model_config(parser)
    return parser


def parse_adaption_config(remaining_argv=None) -> AdaptionConfig:
    parser = get_adaption_parser()
    args = parser.parse_args(remaining_argv)

    # If using YAML file
    if args.config_load_path:
        print(f"Loading config from YAML: {args.config_load_path}")
        yaml_config = load_config_from_yaml(args.config_load_path)

        for key, value in yaml_config.items():
            if hasattr(args, key):
                if not any(opt_str in str(sys.argv) for opt_str in [f'--{key}', f'-{key}']):
                    setattr(args, key, value)

    config = AdaptionConfig(
        num_neighbors=args.num_neighbors,
        root=args.root,
        pretrain_single_graph_data=args.pretrain_single_graph_data,
        num_workers=args.num_workers,
        n_layers=args.n_layers,
        n_flat_layers=args.n_flat_layers,
        in_dim=args.in_dim,
        hid_dim=args.hid_dim,
        fiber_dim=args.fiber_dim,
        att_dim=args.att_dim,
        bias=args.bias,
        act_str=args.act_str,
        drop=args.drop,
        normalize=args.normalize,
        norm_str=args.norm_str,
        temperature=args.temperature,
        gamma=args.gamma,
        loss_reduction=args.loss_reduction,

        data_name=args.data_name,
        pretrained_checkpoint=args.pretrained_checkpoint,
        metric=args.metric,
        task_type=args.task_type,
        k_shot=args.k_shot,
        num_way_link=args.num_way_link,
        num_trials=args.num_trials,
        num_val=args.num_val,
        batch_size=args.batch_size,
        lr_task=args.lr_task,
        task_weight_decay=args.task_weight_decay,
        task_epochs=args.task_epochs,
        max_grad_norm=args.max_grad_norm,
        eval_interval=args.eval_interval,
        resume_checkpoint=args.resume_checkpoint,
        patience=args.patience
    )


    dir_name = ""
    for d in config.pretrain_single_graph_data:
        dir_name += f"{d}_"

    # Paths
    suffix = f"{dir_name[:-1]}/{config.task_type}/{config.k_shot}-shot_{config.data_name}"
    config.log_path = f"logs/{suffix}.log"
    config.checkpoint_dir = f"checkpoints/{suffix}/"

    if args.save_config:
        config_save_path = f"./scripts/{suffix}.yaml"
        save_config_to_yaml(config, config_save_path)

    return config