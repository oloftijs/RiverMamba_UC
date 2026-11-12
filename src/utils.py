

# ------------------------------------------------------------------
"""
Utility functions for SubMamba


"""
# ------------------------------------------------------------------

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

import torch
import numpy as np
import random
import os
import datetime
import logging
from timm.scheduler.step_lr import StepLRScheduler
from timm.scheduler.cosine_lr import CosineLRScheduler
from sklearn.metrics import r2_score
from timm.optim import optim_factory
from types import SimpleNamespace

np.set_printoptions(suppress=True)
torch.set_printoptions(sci_mode=False)

# ------------------------------------------------------------------


def log_string(logger, str):
    logger.info(str)
    print(str)


def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_dir(dir):
    os.makedirs(dir, exist_ok=True)


def get_logger(config):
    # Set Logger and create Directories

    if config.name is None or len(config.name) == 0:
        config.name = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))

    dir_log = os.path.join(config.dir_log, config.name)
    make_dir(dir_log)

    if config.phase == 'train':
        checkpoints_dir = os.path.join(dir_log, 'model_checkpoints/')
        make_dir(checkpoints_dir)

    logger = logging.getLogger("Trainer")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/log_file.txt' % dir_log)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_optimizer(model, config):

    args1 = SimpleNamespace()
    args1.weight_decay = config.weight_decay
    args1.lr = config.lr
    args1.opt = config.optimizer
    args1.momentum = config.beta
    optimizer = optim_factory.create_optimizer(args1, model)

    return optimizer


def get_learning_scheduler(optimizer, config):
    lr_scheduler = config.lr_scheduler

    if lr_scheduler == 'step':
        lr_scheduler = StepLRScheduler(
            optimizer,
            decay_t=config.lr_decay_step,
            decay_rate=config.lr_decay_rate,
            warmup_lr_init=config.lr_warmup,
            warmup_t=config.lr_warmup_epochs,
            t_in_epochs=True,
        )

    elif lr_scheduler == 'cosine':
        lr_scheduler = CosineLRScheduler(
            optimizer,
            t_initial=config.n_epochs,
            cycle_mul=1.,
            lr_min=config.lr_min,
            warmup_lr_init=config.lr_warmup,
            warmup_t=config.lr_warmup_epochs,
            cycle_limit=1,
            t_in_epochs=False,
            warmup_prefix=False
        )

    else:
        raise ValueError('Unexpected optimizer {}, supported scheduler is step or cosine'.format(config.optimizer))

    return lr_scheduler


def r2_score_multi(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """
    Calculated the r-squared score between 2 arrays of values
    """
    index = ~np.logical_or(np.isnan(y_pred), np.isinf(y_pred))
    if len(y_true[index]) == 0:
        return 0.0
    return r2_score(y_true[index], y_pred[index])


class SubwayEvaluator():
    """
    Custom Evaluator for SubMamba.
    Tracks system-wide and line-disaggregated RMSE and Threshold F1-Scores.
    """
    def __init__(self, logger, mode, lead_time: int = 1):
        self.mode = mode
        self.logger = logger
        self.lead_time = lead_time

        # Delays categorized into severity thresholds (in seconds)
        # Minor < 3 mins (180s), Major > 15 mins (900s)
        self.thresholds = [180, 900]
        self.classes = ['Delay > 3 min (Minor/Moderate)', 'Delay > 15 min (Major)']
        self.n_classes = len(self.classes)

        self.reset()

    def reset(self):
        self.seen_iter = 0
        self.total_rmse = 0
        self.total_mae = 0

        # For Threshold F1 System-wide
        self.tp = np.zeros(self.n_classes)
        self.fp = np.zeros(self.n_classes)
        self.fn = np.zeros(self.n_classes)

        # For Spatial Equity (Disaggregated by Subway Line)
        self.line_tp = {}
        self.line_fp = {}
        self.line_fn = {}
        self.line_rmse_sum = {}
        self.line_counts = {}

        # Final Metrics
        self.RMSE = 0.0
        self.F1 = 0.0

    def __call__(self, pred, target, line_ids=None):
        """
        pred: Predicted subway delays in seconds.
        target: Actual subway delays in seconds.
        line_ids: Optional array corresponding to the subway line of each prediction for spatial equity evaluation.
        """
        mse = np.mean((pred - target) ** 2)
        mae = np.mean(np.abs(pred - target))

        self.total_rmse += np.sqrt(mse)
        self.total_mae += mae
        self.seen_iter += 1

        # Threshold classification logic
        for i, thr in enumerate(self.thresholds):
            pred_thr = (pred >= thr)
            target_thr = (target >= thr)

            self.tp[i] += np.sum(pred_thr & target_thr)
            self.fp[i] += np.sum(pred_thr & ~target_thr)
            self.fn[i] += np.sum(~pred_thr & target_thr)

        # Disaggregated Line Logic (Spatial Equity)
        if line_ids is not None:
            unique_lines = np.unique(line_ids)
            for line in unique_lines:
                mask = (line_ids == line)
                if line not in self.line_counts:
                    self.line_counts[line] = 0
                    self.line_rmse_sum[line] = 0.0
                    self.line_tp[line] = np.zeros(self.n_classes)
                    self.line_fp[line] = np.zeros(self.n_classes)
                    self.line_fn[line] = np.zeros(self.n_classes)

                self.line_counts[line] += 1
                self.line_rmse_sum[line] += np.sqrt(np.mean((pred[mask] - target[mask]) ** 2))

                for i, thr in enumerate(self.thresholds):
                    pred_thr = (pred[mask] >= thr)
                    target_thr = (target[mask] >= thr)

                    self.line_tp[line][i] += np.sum(pred_thr & target_thr)
                    self.line_fp[line][i] += np.sum(pred_thr & ~target_thr)
                    self.line_fn[line][i] += np.sum(~pred_thr & target_thr)

    def get_results(self):
        # System-Wide Results
        self.RMSE = self.total_rmse / max(self.seen_iter, 1)
        mae_avg = self.total_mae / max(self.seen_iter, 1)

        precision = self.tp / (self.tp + self.fp + 1e-6)
        recall = self.tp / (self.tp + self.fn + 1e-6)
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-6)
        self.F1 = np.mean(f1_scores)

        message = f'-----------------   {self.mode}   -----------------\n'
        message += f'Overall System -> RMSE: {self.RMSE:.4f}, MAE: {mae_avg:.4f}, Mean Threshold F1: {self.F1:.4f}\n'
        for i, cls in enumerate(self.classes):
            message += f'  {cls} -> F1: {f1_scores[i]:.4f}, Precision: {precision[i]:.4f}, Recall: {recall[i]:.4f}\n'

        # Disaggregated Results (Spatial Equity)
        if self.line_counts:
            message += '\n--- Disaggregated by Subway Line (Spatial Equity) ---\n'
            for line in sorted(self.line_counts.keys()):
                line_rmse = self.line_rmse_sum[line] / self.line_counts[line]

                line_prec = self.line_tp[line] / (self.line_tp[line] + self.line_fp[line] + 1e-6)
                line_rec = self.line_tp[line] / (self.line_tp[line] + self.line_fn[line] + 1e-6)
                line_f1 = 2 * (line_prec * line_rec) / (line_prec + line_rec + 1e-6)

                message += f'Line {line} -> RMSE: {line_rmse:.4f}, F1 (>3m): {line_f1[0]:.4f}, F1 (>15m): {line_f1[1]:.4f}\n'

        log_string(self.logger, message)


def min_max_scale(array, min_array, max_array, min_new=-1., max_new=1.):
    array = ((max_new - min_new) * (array - min_array) / (max_array - min_array)) + min_new
    return array


def save_model(model, optimizer, epoch, mean_loss_train, mean_loss_val, best_loss_train, best_loss_val, logger, config, metric):
    dir_log = os.path.join(config.dir_log, config.name)
    checkpoints_dir = os.path.join(dir_log, 'model_checkpoints/')

    if metric == 'loss':
        path = os.path.join(checkpoints_dir, 'best_loss_model.pth')
        log_string(logger, 'saving model to %s' % path)
    elif metric == 'F1':
        path = os.path.join(checkpoints_dir, 'best_F1_model.pth')
        log_string(logger, 'saving model to %s' % path)
    elif metric == 'RMSE':
        path = os.path.join(checkpoints_dir, 'best_RMSE_model.pth')
        log_string(logger, 'saving model to %s' % path)
    elif metric == 'train':
        path = os.path.join(checkpoints_dir, 'best_train_model.pth')
    else:
        path = os.path.join(checkpoints_dir, f'model_{metric}.pth')

    state = {
        'epoch': epoch,
        'mean_loss_train': mean_loss_train,
        'mean_loss_validation': mean_loss_val,
        'best_loss_train': best_loss_train,
        'best_loss_validation': best_loss_val,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }

    torch.save(state, path)
