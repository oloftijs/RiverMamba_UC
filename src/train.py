# ------------------------------------------------------------------
"""
Main script for training and validating SubMamba (Adapted from RiverMamba)
"""
# ------------------------------------------------------------------

import os
import torch
import numpy as np
from tqdm import tqdm
from torch import autocast
from torch.utils.tensorboard import SummaryWriter

# SubMamba Modules
import config as config_file
from dataset.dataset import SubwayDelayDataset
from model.Mamba2_encoder import Mamba2 as EncoderMamba2
from model.Mamba2_decoder import Mamba2 as DecoderMamba2
from model.MLP_head import MLP as RegressionHead

np.set_printoptions(suppress=True)
torch.set_printoptions(sci_mode=False)
np.seterr(divide='ignore', invalid='ignore')


class Model(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = EncoderMamba2(
            in_dynamic=config.in_dynamic,
            in_static=config.in_static,
            embed_dynamic=config.en_embed_dim[0],
            embed_dim=config.en_embed_dim,
            depths=config.en_depths,
            grouping_size=config.en_grouping_size
        )
        self.decoder = DecoderMamba2(
            in_features=config.en_embed_dim[-1],
            in_static=config.in_static,
            delta_t_f=config.delta_t_f,
            embed_dim=config.de_embed_dim,
            depths=config.de_depths,
            grouping_size=config.de_grouping_size
        )
        self.head = RegressionHead(
            embed_dim=config.de_embed_dim[-1],
            embed_dim_all=config.de_embed_dim[-1] * config.delta_t_f,
            hidden_dim=config.head_hidden_dim,
            out_dim=config.delta_t_f
        )
        self.criterion = torch.nn.MSELoss()

    def forward(self, x_dynamic, x_static, graph_order, y_target=None):
        # 1. Encode: [B, T, P, 192]
        enc_out = self.encoder(x_dynamic, x_static, graph_order)

        # 2. Slice last timestep to match DecoderMamba2's expected [B, 1, P, K] input
        enc_last = enc_out[:, -1:, :, :]

        # 3. Decode: returns list of delta_t_f tensors, each [B, 1, P, 192]
        x_future = self.decoder(enc_last, x_static, graph_order)

        # 4. Predict lead time sequence
        pred = self.head(enc_last, x_future)  # [B, 1, P, delta_t_f]
        pred = pred.squeeze(1)                # [B, P, delta_t_f]

        loss = None
        if y_target is not None:
            loss = self.criterion(pred, y_target)

        return pred, loss


class DummyLogger:
    def info(self, msg):
        print(msg)


def log_string(logger, msg):
    logger.info(msg)


def get_optimizer(model, config):
    return torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)


def get_learning_scheduler(optimizer, config):
    return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.n_epochs, eta_min=config.lr_min)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_checkpoint(state, filepath):
    torch.save(state, filepath)


def compute_val_baselines(train_dataloader, val_dataloader):
    """
    Computes exact element-wise MSE, RMSE, and ME (pred - target) on the validation
    tensors for three baselines so they are directly comparable to model validation metrics:
      1. zero         : predicts 0.0 everywhere
      2. persistence  : repeats the last historical bin delay (x_dynamic[:, -1, :, 0])
      3. station_mean : predicts each station's mean delay from the training split
    """
    # 1. Compute per-station mean target delay from training set
    train_sum = None
    train_count = 0
    for data_i in train_dataloader:
        y = data_i['y_target'].double()  # [B, P, delta_t_f]
        if train_sum is None:
            train_sum = torch.zeros(y.shape[1], dtype=torch.float64)
        train_sum += y.sum(dim=(0, 2))
        train_count += y.shape[0] * y.shape[2]

    station_mean = (train_sum / max(1, train_count)).float() if train_sum is not None else torch.zeros(1)

    # 2. Evaluate baselines across validation set
    stats = {
        'zero': {'se': 0.0, 'err': 0.0},
        'persistence': {'se': 0.0, 'err': 0.0},
        'station_mean': {'se': 0.0, 'err': 0.0},
    }
    total_elems = 0

    for data_i in val_dataloader:
        target = data_i['y_target'].double()                  # [B, P, delta_t_f]
        last_delay = data_i['x_dynamic'][:, -1, :, 0].double()  # [B, P]

        preds = {
            'zero': torch.zeros_like(target),
            'persistence': last_delay.unsqueeze(-1).expand_as(target),
            'station_mean': station_mean.double().view(1, -1, 1).expand_as(target),
        }

        for name, pred in preds.items():
            diff = pred - target
            stats[name]['se'] += (diff ** 2).sum().item()
            stats[name]['err'] += diff.sum().item()

        total_elems += target.numel()

    denom = max(1, total_elems)
    results = {}
    for name, s in stats.items():
        mse = s['se'] / denom
        results[name] = {
            'mse': mse,
            'rmse': float(np.sqrt(mse)),
            'mae': float(np.abs(s['err']) / denom),
        }
    return results


def print_baselines(logger, baselines):
    for name, m in baselines.items():
        log_string(
            logger,
            f'Baseline [{name:<12}] MSE: {m["mse"]:.6f} | RMSE: {m["rmse"]:.2f}s | MAE: {m["mae"]:+.2f}s'
        )


def train():
    config = config_file.read_arguments(train=True, print=False, save=True)
    logger = DummyLogger()

    exp_dir = os.path.join(config.dir_log, config.name)
    os.makedirs(exp_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=exp_dir)

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    # Dataloaders
    log_string(logger, "loading training and validation datasets ...")
    train_dataset = SubwayDelayDataset(
        tensors_dir='tensors_out',
        split='train',
        delta_t=config.delta_t,
        delta_t_f=config.delta_t_f,
        n_points=config.n_points,
        graph_order=config.graph_vertex_order
    )
    val_dataset = SubwayDelayDataset(
        tensors_dir='tensors_out',
        split='val',
        delta_t=config.delta_t,
        delta_t_f=config.delta_t_f,
        n_points=config.n_points,
        graph_order=config.graph_vertex_order
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.batch_size_train,
        shuffle=True,
        pin_memory=config.pin_memory,
        num_workers=0
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config.batch_size_val,
        shuffle=False,
        pin_memory=config.pin_memory,
        num_workers=0
    )

    log_string(logger, f"# training samples  : {len(train_dataset)}")
    log_string(logger, f"# validation samples: {len(val_dataset)}")

    # Compute validation baselines before starting training
    log_string(logger, "\ncomputing validation baselines ...")
    val_baselines = compute_val_baselines(train_dataloader, val_dataloader)
    print_baselines(logger, val_baselines)

    # Model setup
    log_string(logger, "\nloading the model ...")
    device = torch.device('cuda' if torch.cuda.is_available() and config.gpu_id != "-1" else 'cpu')
    model = Model(config).to(device)

    log_string(logger, "model parameters ...")
    log_string(logger, f"all parameters: {count_parameters(model)}\n")

    optimizer = get_optimizer(model, config)
    lr_scheduler = get_learning_scheduler(optimizer, config)

    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))
    max_norm = config.max_norm
    amp_dtype = torch.bfloat16 if device.type == 'cuda' else torch.float32

    best_loss_val = float('inf')

    log_string(logger, 'training on NYC Subway GTFS dataset ...\n')

    for epoch in range(config.n_epochs):
        log_string(logger, f'################# Epoch ({epoch + 1}/{config.n_epochs}) #################')

        # --- Training ---
        model.train()
        se_train = 0.0
        err_train = 0.0
        elems_train = 0

        pbar_train = tqdm(enumerate(train_dataloader), total=len(train_dataloader), smoothing=0.9, postfix="  training")
        for i, data_i in pbar_train:
            optimizer.zero_grad(set_to_none=True)
            target = data_i['y_target'].to(device)

            with autocast(device_type=device.type, dtype=amp_dtype):
                pred, loss = model(
                    data_i['x_dynamic'].to(device),
                    data_i['x_static'].to(device),
                    data_i['graph_vertex_order'].to(device).long(),
                    target
                )
                loss = torch.nan_to_num(loss)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            scaler.step(optimizer)
            scaler.update()

            diff = torch.nan_to_num(pred.detach().double() - target.double())
            se_train += (diff ** 2).sum().item()
            err_train += diff.sum().item()
            elems_train += target.numel()
            abs_diff = torch.abs(diff)
            batch_me = abs_diff.mean().item()
            pbar_train.set_description(f'train loss {loss.item():.4f} | MSE {batch_me:+.2f}s', refresh=True)

        lr_scheduler.step()
        denom_train = max(1, elems_train)
        mean_loss_train = se_train / denom_train
        mean_me_train = err_train / denom_train
        log_string(
            logger,
            f'Training   mean MSE: {mean_loss_train:.6f} | RMSE: {np.sqrt(mean_loss_train):.2f}s | ME: {mean_me_train:+.2f}s'
        )

        # --- Validation ---
        model.eval()
        se_val = 0.0
        err_val = 0.0
        elems_val = 0

        with torch.no_grad():
            pbar_val = tqdm(enumerate(val_dataloader), total=len(val_dataloader), smoothing=0.9, postfix="  validation")
            for i, data_i in pbar_val:
                target = data_i['y_target'].to(device)

                with autocast(device_type=device.type, dtype=amp_dtype):
                    pred, loss = model(
                        data_i['x_dynamic'].to(device),
                        data_i['x_static'].to(device),
                        data_i['graph_vertex_order'].to(device).long(),
                        target
                    )
                    loss = torch.nan_to_num(loss)

                diff = torch.nan_to_num(pred.double() - target.double())
                se_val += (diff ** 2).sum().item()
                err_val += diff.sum().item()
                elems_val += target.numel()

                batch_me = diff.mean().item()
                pbar_val.set_description(f'val loss {loss.item():.4f} | ME {batch_me:+.2f}s', refresh=True)

        denom_val = max(1, elems_val)
        mean_loss_val = se_val / denom_val
        mean_me_val = np.abs(err_val) / denom_val
        log_string(
            logger,
            f'Validation mean MSE: {mean_loss_val:.6f} | RMSE: {np.sqrt(mean_loss_val):.2f}s | MAE: {mean_me_val:+.2f}s'
        )
        print_baselines(logger, val_baselines)

        writer.add_scalars("Loss/MSE", {'train': mean_loss_train, 'val': mean_loss_val}, epoch + 1)
        writer.add_scalars("Error/MAE", {'train': mean_me_train, 'val': mean_me_val}, epoch + 1)
        writer.add_scalar("LR", optimizer.param_groups[0]['lr'], epoch + 1)

        # --- Save Checkpoints ---
        checkpoint_state = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': lr_scheduler.state_dict(),
            'train_loss': mean_loss_train,
            'train_me': mean_me_train,
            'val_loss': mean_loss_val,
            'val_me': mean_me_val,
            'best_val_loss': min(best_loss_val, mean_loss_val),
            'val_baselines': val_baselines,
        }

        save_checkpoint(checkpoint_state, os.path.join(exp_dir, 'last_checkpoint.pth'))

        if mean_loss_val < best_loss_val:
            best_loss_val = mean_loss_val
            save_checkpoint(checkpoint_state, os.path.join(exp_dir, 'best_model.pth'))
            log_string(
                logger,
                f'--> Saved new best model (val MSE: {best_loss_val:.6f} | RMSE: {np.sqrt(best_loss_val):.2f}s | MAE: {mean_me_val:+.2f}s)'
            )

        log_string(logger, '')

    writer.close()


if __name__ == '__main__':
    train()