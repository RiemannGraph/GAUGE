import torch
import torch.nn.functional as F
import numpy as np
from abc import ABC, abstractmethod
from sklearn.metrics import roc_auc_score


class BaseTask(ABC):
    def __init__(self, device, metric="acc"):
        self.device = device
        self.metric = metric

    @abstractmethod
    def get_label_key(self):
        """Return the attribute name of labels in data object."""
        pass

    @abstractmethod
    def compute_loss(self, pred, label):
        """Compute task-specific loss (excluding model regularization)."""
        pass

    @abstractmethod
    def get_prediction(self, pred):
        """Convert raw model output to prediction (e.g., prob or class)."""
        pass

    @abstractmethod
    def compute_metric(self, preds, trues):
        """Compute evaluation metric."""
        pass

    def get_compute_instance(self, preds, trues, batch):
        return preds, trues

    def train_step(self, loader, optimizer, model):
        """
        Generic training step.
        Args:
            loader: DataLoader or iterable of batched data.
            optimizer: torch.optim.Optimizer.
            model: nn.Module.
        """
        model.train()
        total_loss = 0.0
        all_preds = []
        all_trues = []

        for batch in loader:
            batch = batch.to(self.device)
            optimizer.zero_grad()
            pred, aux_loss = model(batch)

            label = getattr(batch, self.get_label_key())
            pred, label = self.get_compute_instance(pred, label, batch)

            loss = self.compute_loss(pred, label) + aux_loss
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            all_preds.append(self.get_prediction(pred).detach().cpu().numpy())
            all_trues.append(label.detach().cpu().numpy())

        avg_loss = total_loss / len(loader)
        preds = np.concatenate(all_preds, axis=0)
        trues = np.concatenate(all_trues, axis=0)
        metric_val = self.compute_metric(preds, trues)
        return avg_loss, metric_val

    @torch.no_grad()
    def eval_step(self, loader, model):
        model.eval()
        total_loss = 0.0
        all_preds = []
        all_trues = []

        for batch in loader:
            batch = batch.to(self.device)
            pred, aux_loss = model(batch)

            label = getattr(batch, self.get_label_key())
            pred, label = self.get_compute_instance(pred, label, batch)

            loss = self.compute_loss(pred, label) + aux_loss
            total_loss += loss.item()
            all_preds.append(self.get_prediction(pred).cpu().numpy())
            all_trues.append(label.cpu().numpy())

        avg_loss = total_loss / len(loader)
        preds = np.concatenate(all_preds, axis=0)
        trues = np.concatenate(all_trues, axis=0)
        metric_val = self.compute_metric(preds, trues)
        return avg_loss, metric_val


class GraphClassificationTask(BaseTask):
    def get_label_key(self):
        return 'y'

    def compute_loss(self, pred, label):
        return F.cross_entropy(pred, label)

    def get_prediction(self, pred):
        return pred.argmax(dim=-1)

    def compute_metric(self, preds, trues):
        if self.metric == "acc":
            return (preds == trues).mean()
        elif self.metric == "auc":
            return roc_auc_score(trues, preds)
        else:
            raise ValueError(f"Unsupported metric for classification: {self.metric}")


class NodeClassificationTask(GraphClassificationTask):
    def get_compute_instance(self, preds, trues, batch):
        return preds[: batch.batch_size], trues[: batch.batch_size]


class LinkPredictionTask(BaseTask):
    def get_label_key(self):
        return 'edge_label'

    def compute_loss(self, pred, label):
        # pred: [E] or [E, 1]; label: [E] with 0/1
        if pred.dim() > 1 and pred.size(1) == 1:
            pred = pred.squeeze(1)
        return F.binary_cross_entropy_with_logits(pred, label.float())

    def get_prediction(self, pred):
        if pred.dim() > 1 and pred.size(1) == 1:
            pred = pred.squeeze(1)
        return pred

    def compute_metric(self, preds, trues):
        trues = trues.astype(int)
        if self.metric == "auc":
            return roc_auc_score(trues, preds)
        else:
            raise ValueError(f"Unsupported metric for link prediction: {self.metric}")