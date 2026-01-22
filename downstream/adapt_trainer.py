import os
import time

import numpy as np
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_geometric.loader import NeighborLoader, LinkNeighborLoader

from cores.models import Characteron
from data import (
    load_few_shot_single_graph_data,
    load_link_graph_data
)
from downstream.adapter import CharacteronAdapter
from downstream.tasks import NodeClassificationTask, GraphClassificationTask, LinkPredictionTask
from utils.checkpoints import (
    load_checkpoint,
    EarlyStopping
)
from utils.logger import create_logger

TASK_REGISTRY = {
    'node_cls': NodeClassificationTask,
    'graph_cls': GraphClassificationTask,
    'link_cls': LinkPredictionTask,
}


class AdaptTrainer:

    def __init__(self, configs, logger=None):
        self.configs = configs
        self.device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
        self.logger = logger if logger is not None else create_logger(configs.log_path)
        self.task_handler = TASK_REGISTRY[configs.task_type](device=self.device, metric=configs.metric)

        self.start_epoch = 0

        self.task_type = configs.task_type

        os.makedirs(self.configs.checkpoint_dir, exist_ok=True)
        os.makedirs("./results", exist_ok=True)

    def train(self):
        loaders, num_classes, num_features = self.get_loaders(self.configs)
        train_loaders = loaders[0]
        val_loaders = loaders[1]
        test_loaders = loaders[2]

        # Train loop
        total_metric = []
        total_test_loss = []
        with open(f"./results/{self.configs.data_name}.txt", "a") as f:
            f.write(f"============={self.configs.k_shot}-Shot {self.configs.task_type}=================\n")
            f.write(f"Pretraining Model: {self.configs.pretrained_checkpoint}\n")
        f.close()
        for trial in range(self.configs.num_trials):
            pretrained_model = Characteron(self.configs)
            load_checkpoint(self.configs.pretrained_checkpoint, pretrained_model, map_location='cuda')
            model = CharacteronAdapter(self.configs, num_features, pretrained_model,
                                        self.configs.task_type, num_classes).to(self.device)
            optimizer = Adam(
                model.parameters(),
                lr=self.configs.lr_task,
                weight_decay=self.configs.task_weight_decay
            )
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=self.configs.task_epochs,
                eta_min=self.configs.lr_task * 0.01
            )
            early_stopping = EarlyStopping(
                patience=self.configs.patience,
                mode='max',
                delta=0.001,
                checkpoint_dir=self.configs.checkpoint_dir,
                verbose=True
            )
            if self.configs.k_shot > 0:
                model.train()
                for epoch in range(self.start_epoch, self.configs.task_epochs):
                    epoch_start_time = time.time()
                    train_loss, train_metric = self._train_epoch(train_loaders[trial], model, optimizer, trial)
                    scheduler.step()
                    epoch_time = time.time() - epoch_start_time

                    self.logger.info(
                        f'Epoch {epoch:03d}/{self.configs.task_epochs} | '
                        f'Train Loss: {train_loss:.6f} | '
                        f'Train {self.configs.metric.upper()}: {train_metric * 100:.2f}% | '
                        f'Time: {epoch_time:.2f}s | '
                        f'LR: {optimizer.param_groups[0]["lr"]:.2e}'
                    )

                    # Evaluation
                    if (epoch + 1) % self.configs.eval_interval == 0:
                        val_loss, val_metric = self.task_handler.eval_step(val_loaders[trial], model)
                        self.logger.info(f'Epoch {epoch:03d} | Val {self.configs.metric.upper()}: {val_metric * 100:.2f}%')

                        if early_stopping.step(
                                metric=val_metric,
                                model=model,
                                optimizer=optimizer,
                                scheduler=scheduler,
                                epoch=epoch,
                                config=self.configs
                        ):
                            break

                # Final save
                final_path = os.path.join(self.configs.checkpoint_dir, f'downstream_final_{trial}.pth')
                torch.save({'state_dict': model.state_dict()}, final_path)
                self.logger.info(f"Trial {trial} | Training finished. Final model saved to {final_path}")

                self.logger.info(f"===========Loading best checkpoint from {self.configs.checkpoint_dir}/model_best.pth===========")
                load_checkpoint(f"{self.configs.checkpoint_dir}/model_best.pth", model)
            model.eval()
            test_loss, test_metric = self.task_handler.eval_step(test_loaders[trial], model)
            self.logger.info("=====================================================")
            info = f'Trial {trial:02d} | Test {self.configs.metric.upper()}: {test_metric * 100:.2f}%' \
                             f'| Test Loss: {test_loss:.6f} '
            self.logger.info(info)
            self.logger.info("=====================================================")
            total_metric.append(test_metric)
            total_test_loss.append(test_loss)
            with open(f"./results/{self.configs.data_name}.txt", "a") as f:
                f.write(info + "\n")
            f.close()
        info = f'Final Test {self.configs.metric.upper()}: ' \
                f'{np.mean(total_metric) * 100:.2f} \u00B1 {np.std(total_metric) * 100:.2f} % \n' \
                f'Final Test Loss: {np.mean(total_test_loss):.6f} \u00B1 {np.std(total_test_loss):.6f} \n'
        self.logger.info(info)
        with open(f"./results/{self.configs.data_name}.txt", "a") as f:
            f.write(info + "\n")
            f.write("======================================================================\n")
        f.close()

    def _train_epoch(self, train_loader, model, optimizer, trial):
        loss, acc = self.task_handler.train_step(train_loader, optimizer, model)
        return loss, acc

    def get_loaders(self, configs):
        train_loaders = []
        val_loaders = []
        test_loaders = []
        if configs.task_type == "node_cls":
            dataset, data, train_mask, val_mask, test_mask = load_few_shot_single_graph_data(configs, configs.data_name,
                                                                                      configs.k_shot,
                                                                                      configs.num_trials,
                                                                                      configs.num_val)
            num_classes = dataset.num_classes
            num_features = dataset.num_features
            for t in range(configs.num_trials):
                train_loaders.append(NeighborLoader(data, input_nodes=train_mask[:, t],
                                                batch_size=configs.batch_size,
                                                shuffle=True, num_neighbors=configs.num_neighbors
                                                ))
                val_loaders.append(NeighborLoader(data, input_nodes=val_mask[:, t],
                                              batch_size=configs.batch_size,
                                              shuffle=False, num_neighbors=configs.num_neighbors
                                              ))
                test_loaders.append(NeighborLoader(data, input_nodes=test_mask[:, t],
                                               batch_size=configs.batch_size,
                                               shuffle=False, num_neighbors=configs.num_neighbors
                                               ))
        # elif configs.task_type == "graph_cls":
        #     dataset, train_mask, val_mask, test_mask = load_few_shot_multi_graph_data(configs, configs.data_name,
        #                                                    configs.k_shot, configs.num_trials,
        #                                                    configs.num_val)
        #     num_classes = dataset.num_classes
        #     num_features = dataset.num_features
        #     for t in range(configs.num_trials):
        #         train_loaders.append(DataLoader(dataset[train_mask[:, t]],
        #                                         batch_size=configs.batch_size,
        #                                         shuffle=True))
        #         val_loaders.append(DataLoader(dataset[val_mask[:, t]],
        #                                       batch_size=configs.batch_size,
        #                                       shuffle=False))
        #         test_loaders.append(DataLoader(dataset[test_mask[:, t]],
        #                                        batch_size=configs.batch_size,
        #                                        shuffle=False))
        #
        elif configs.task_type == "link_cls":
            for t in range(configs.num_trials):
                dataset, train_data, val_data, test_data = load_link_graph_data(configs, configs.data_name)
                num_classes = None
                num_features = dataset.num_features
                train_loaders.append(LinkNeighborLoader(train_data,
                                                    batch_size=configs.batch_size,
                                                    num_neighbors=self.configs.num_neighbors,
                                                    edge_label_index=train_data.edge_label_index,
                                                    edge_label=train_data.edge_label,
                                                    shuffle=True))
                val_loaders.append(LinkNeighborLoader(val_data,
                                                    batch_size=configs.batch_size,
                                                    num_neighbors=self.configs.num_neighbors,
                                                    edge_label_index=val_data.edge_label_index,
                                                    edge_label=val_data.edge_label,
                                                    shuffle=True))
                test_loaders.append(LinkNeighborLoader(test_data,
                                                    batch_size=configs.batch_size,
                                                    num_neighbors=self.configs.num_neighbors,
                                                    edge_label_index=test_data.edge_label_index,
                                                    edge_label=test_data.edge_label,
                                                    shuffle=True))
        else:
            raise NotImplementedError
        return (train_loaders, val_loaders, test_loaders), num_classes, num_features