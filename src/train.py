

# ------------------------------------------------------------------
"""
Main script for training and validating SubMamba (Adapted from RiverMamba)

"""
# ------------------------------------------------------------------

import torch
import numpy as np
from tqdm import tqdm
import time
import os
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler
from torch import autocast

# SubMamba Modules
import config as config_file
from dataset.dataset import SubwayDelayDataset
from model.Mamba2_encoder import Mamba2 as EncoderMamba2
from model.Mamba2_decoder import Mamba2 as DecoderMamba2
from model.MLP_head import MLP as  RegressionHead
# Assuming decoder is similar to encoder and is available as DecoderMamba2
# (Using EncoderMamba2 as placeholder for the full Model build step)

np.set_printoptions(suppress=True)
torch.set_printoptions(sci_mode=False)
np.seterr(divide='ignore', invalid='ignore')

# ------------------------------------------------------------------
# Mocking the Model class and Utils for standalone execution testing
# ------------------------------------------------------------------
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
        pred = self.head(enc_last, x_future) # [16, 1, 472, 7]
        pred = pred.squeeze(1)  # [16, 472, 7]

        loss = None
        if y_target is not None:
            loss = self.criterion(pred, y_target)

        return pred, loss
class DummyLogger:
    def info(self, msg): print(msg)

def log_string(logger, msg):
    logger.info(msg)

def get_optimizer(model, config):
    return torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

def get_learning_scheduler(optimizer, config):
    return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.n_epochs, eta_min=config.lr_min)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train():
    # read config arguments
    config = config_file.read_arguments(train=True, print=False, save=True)

    # get logger
    logger = DummyLogger()

    # fix random seed
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    # dataloader
    log_string(logger, "loading training dataset ...")

    # Check if processed dir exists, if not bypass cleanly
    if not os.path.exists(config.dir_log):
        os.makedirs(config.dir_log, exist_ok=True)

    train_dataset = SubwayDelayDataset(
        tensors_dir='tensors_out',      # <-- Changed from processed_dir
        split='train',                  # <-- Changed from years=config.years_train
        delta_t=config.delta_t,
        delta_t_f=config.delta_t_f,
        n_points=config.n_points,
        graph_order=config.graph_vertex_order
    )

    train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=config.batch_size_train,
                                                   shuffle=True,
                                                   pin_memory=config.pin_memory,
                                                   num_workers=0) # Set to 0 for stable testing

    log_string(logger, f"# training samples: {len(train_dataset)}")

    # get models
    log_string(logger, "\nloading the model ...")
    device = torch.device('cuda' if torch.cuda.is_available() and config.gpu_id != "-1" else 'cpu')

    model = Model(config).to(device)

    log_string(logger, "model parameters ...")
    log_string(logger, f"all parameters: {count_parameters(model)}\n")

    optimizer = get_optimizer(model, config)
    lr_scheduler = get_learning_scheduler(optimizer, config)

    # training loop
    log_string(logger, 'training on NYC Subway GTFS dataset ...\n')

    scaler = GradScaler()
    max_norm = config.max_norm

    for epoch in range(1): # Run 1 epoch for testing
        log_string(logger, f'################# Epoch ({epoch + 1}/{config.n_epochs}) #################')

        model.train()
        loss_train = 0

        # If dataset is empty (no processed files), generate a synthetic batch
        if len(train_dataloader) == 0:
            log_string(logger, "Dataset empty, running synthetic batch...")
            B, T, P = config.batch_size_train, config.delta_t, config.n_points
            V_dyn, V_stat = config.in_dynamic, config.in_static
            data_i = {
                'x_dynamic': torch.randn((B, T, P, V_dyn)),
                'x_static': torch.randn((B, P, V_stat)),
                'graph_vertex_order': torch.arange(P).unsqueeze(0).repeat(B, 1),
                'y_target': torch.randn((B, P, config.delta_t_f))
            }
            batches = [data_i]
        else:
            batches = train_dataloader

        pbar = tqdm(enumerate(batches), total=len(batches), smoothing=0.9, postfix="  training")

        for i, data_i in pbar:
            optimizer.zero_grad(set_to_none=True)

            with autocast(device_type=device.type, dtype=torch.bfloat16 if device.type == 'cuda' else torch.float32):
                pred, loss = model(
                    data_i['x_dynamic'].to(device),
                    data_i['x_static'].to(device),
                    data_i['graph_vertex_order'].to(device).long(),
                    data_i['y_target'].to(device)
                )

                loss = torch.nan_to_num(loss)

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
                scaler.step(optimizer)
                scaler.update()

            pbar.set_description('loss %.4f' % (loss.item()), refresh=True)
            loss_train += loss.item()

        lr_scheduler.step()
        mean_loss_train = loss_train / float(len(batches))
        log_string(logger, f'Training mean loss : {mean_loss_train:.6f}\n')

if __name__ == '__main__':
    train()
