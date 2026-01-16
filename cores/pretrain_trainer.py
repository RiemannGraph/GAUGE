import torch
from torch_geometric.loader import NeighborLoader
from cores.models import GraphTrivializeModel
from data import load_pretrain_single_graph_data
from utils import (
    save_checkpoint,
    load_checkpoint,
    get_latest_checkpoint,
    cleanup_old_checkpoints,
    create_logger,
    format_time)
import os
import time
import itertools
import tqdm
import warnings

warnings.filterwarnings("ignore")


class Pretrainer:
    def __init__(self, configs, logger=None):
        self.final_model_path = None
        self.configs = configs
        self.pretrain_single_graph_data = configs.pretrain_single_graph_data
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = GraphTrivializeModel(configs).to(self.device)
        self.logger = create_logger(configs.log_path) if logger is None else logger
        self.start_epoch = 0
        self.start_time = None
        self.epoch_times = []

        os.makedirs(self.configs.checkpoint_dir, exist_ok=True)

    def train(self):
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.configs.lr_pretrain,
            weight_decay=self.configs.pretrain_weight_decay
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.configs.pretrain_epochs,
            eta_min=self.configs.lr_pretrain * 0.01
        )

        if self.configs.resume_checkpoint:
            latest_check_path = get_latest_checkpoint(self.configs.checkpoint_dir)
            if latest_check_path:
                self.start_epoch = load_checkpoint(latest_check_path, self.model, optimizer, scheduler)
                self.logger.info(f"Resumed from main checkpoint at epoch {self.start_epoch}")
            else:
                self.start_epoch = 0
        else:
            self.start_epoch = 0

        for epoch in range(self.start_epoch, self.configs.pretrain_epochs):
            epoch_start_time = time.time()

            train_loss = self._train_epoch(optimizer, epoch)

            scheduler.step()

            epoch_time = time.time() - epoch_start_time
            self.logger.info(
                f'Epoch {epoch:03d}/{self.configs.pretrain_epochs} | '
                f'Train Loss: {train_loss:.6f} | '
                f'Time: {epoch_time:.2f}s | '
                f'LR: {optimizer.param_groups[0]["lr"]:.2e}'
            )

            if (epoch + 1) % self.configs.save_interval == 0 or (epoch + 1) == self.configs.pretrain_epochs:
                checkpoint_path = os.path.join(
                    self.configs.checkpoint_dir,
                    f'pretrain_epoch_{epoch + 1}.pth'
                )
                save_checkpoint(
                    model=self.model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=epoch + 1,
                    config=self.configs.__dict__,
                    filepath=checkpoint_path
                )

                # Optional
                cleanup_old_checkpoints(self.configs.checkpoint_dir, keep_last=20)

            if (epoch + 1) == self.configs.pretrain_epochs:
                final_model_path = os.path.join(
                    self.configs.checkpoint_dir,
                    'pretrain_final_model.pth'
                )
                save_checkpoint(
                    model=self.model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=epoch + 1,
                    config=self.configs.__dict__,
                    filepath=final_model_path
                )
                self.logger.info(f'Saved final model: {final_model_path}')
                self.final_model_path = final_model_path

    def _train_epoch(self, optimizer, epoch):
        if self.start_time is None:
            self.start_time = time.time()
        start_epoch_time = time.time()

        self.model.train()
        total_loss = 0.0
        total_steps = 0

        loaders = self._get_all_loaders()
        loader_iters = [iter(itertools.cycle(loader)) for loader in loaders]

        # 1. max loader length
        max_steps = max(len(loader) for loader in loaders)
        # 2：weighted w.r.t. num_nodes
        # total_nodes = sum([data.num_nodes for data in all_data_objects])
        # weights = [data.num_nodes / total_nodes for data in all_data_objects]
        # max_steps = int(sum(len(loader) for loader in loaders) / len(loaders))

        for step in tqdm.tqdm(range(max_steps)):
            optimizer.zero_grad()
            step_loss = 0.0

            for loader_iter in loader_iters:
                data = next(loader_iter).to(self.device)
                z, frame = self.model(data)
                loss_i = self.model.loss(z, frame, data, data.batch_size)
                step_loss += loss_i

            step_loss.backward()
            optimizer.step()

            total_loss += step_loss.item()
            total_steps += 1

            if (step + 1) % self.configs.log_interval == 0:
                self._log_progress(
                    epoch=epoch,
                    batch_idx=step + 1,
                    dataset_len=max_steps,
                    loss=step_loss.item(),
                    start_loader_time=time.time(),
                    batches_done=step + 1
                )

        # Log
        self._log_epoch_summary(epoch, start_epoch_time)
        self._update_epoch_time(epoch, start_epoch_time)

        return total_loss / max(1, total_steps)

    def _log_progress(self, epoch, batch_idx, dataset_len, loss, start_loader_time,
                      batches_done):
        current_time = time.time()

        batches_remaining = dataset_len - batches_done
        recent_avg_batch_time = (current_time - start_loader_time) / batches_done
        loader_remaining_time = recent_avg_batch_time * batches_remaining

        if len(self.epoch_times) == 0:
            elapsed_total = current_time - self.start_time
            avg_epoch_time = elapsed_total / (epoch + 1)
        else:
            avg_epoch_time = sum(self.epoch_times) / len(self.epoch_times)

        remaining_epochs = max(0, self.configs.pretrain_epochs - (epoch + 1))

        if epoch == 0:
            total_remaining_time = None
        else:
            total_remaining_time = avg_epoch_time * remaining_epochs

        self.logger.info(
            f'Epoch {epoch} | Batch {batch_idx}/{dataset_len} | '
            f'Loss: {loss:.6f} | '
            f'Loader ETA: {format_time(loader_remaining_time)} | '
            f'Total ETA: {format_time(total_remaining_time)}'
        )

    def _log_epoch_summary(self, epoch, start_epoch_time):
        if len(self.epoch_times) == 0:
            avg_epoch_time = time.time() - self.start_time
        else:
            avg_epoch_time = sum(self.epoch_times) / len(self.epoch_times)

        remaining_epochs = max(0, self.configs.pretrain_epochs - (epoch + 1))
        if epoch == 0:
            total_remaining_time = None
        else:
            total_remaining_time = avg_epoch_time * remaining_epochs

        epoch_duration = time.time() - start_epoch_time

        self.logger.info(
            f'Epoch {epoch} completed in {format_time(epoch_duration)}. '
            f'Estimated remaining training time: {format_time(total_remaining_time)} '
            f'({remaining_epochs} epochs left)'
        )

    def _update_epoch_time(self, epoch, start_epoch_time):
        epoch_duration = time.time() - start_epoch_time
        self.epoch_times.append(epoch_duration)

    def _get_all_loaders(self):
        loaders = []
        for data_name in self.configs.pretrain_single_graph_data:
            data = load_pretrain_single_graph_data(self.configs, data_name)

            loader = NeighborLoader(data, input_nodes=None, batch_size=self.configs.batch_size, num_neighbors=self.configs.num_neighbors,
                            num_workers=self.configs.num_workers, persistent_workers=False)
            loaders.append(loader)
        return loaders