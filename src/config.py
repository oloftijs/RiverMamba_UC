

# ------------------------------------------------------------------
"""
Main config file for SubMamba (Adapted from RiverMamba)
"""
# ------------------------------------------------------------------

import argparse
import pickle
import os
import datetime

# ------------------------------------------------------------------


def add_all_arguments(parser):

    # --- general options --- #
    parser.add_argument('--name', type=str, default='test_submamba', help='name of the experiment')
    parser.add_argument('--dir_log', type=str, default=r'./log', help='log folder')

    parser.add_argument('--encoder', type=str, default='Mamba2', help='name of the encoder model')
    parser.add_argument('--decoder', type=str, default='Mamba2', help='name of the decoder model')
    parser.add_argument('--head', type=str, default='MLP', help='name of the regression head model')

    # --- encoder --- #
    parser.add_argument('--en_embed_dynamic', type=int, default=64, help='embedding dimension for dynamic subway state')
    parser.add_argument('--en_embed_dim', type=int, default=[192, 192, 192],
                        help='embedding dimensions in the encoder model')
    parser.add_argument('--en_depths', type=int, default=[2, 2, 2], help='number of blocks inside each layer')

    # Adjust grouping size according to typical subway graph dimensions
    parser.add_argument('--en_grouping_size', type=int, default=[(4, 472), (2, 472), (1, 472)],
                        help='grouping size for mamba/self-attention')
    parser.add_argument('--en_mlp_ratio', type=float, default=1., help='ratio of mlp hidden dim to embedding dim')
    parser.add_argument('--en_drop_rate', type=float, default=0., help='dropout rate')
    parser.add_argument('--en_drop_path_rate', type=float, default=0., help='stochastic depth rate')
    parser.add_argument('--en_embed_norm', type=bool, default=True,
                        help='if True, add normalization after point embedding')
    parser.add_argument('--en_use_checkpoint', type=bool, default=False,
                        help='whether to use checkpointing to save memory')
    parser.add_argument('--en_use_reentrant', type=bool, default=True, help='whether to use reentrant for checkpoint')

    # encoder Mamba
    parser.add_argument('--en_d_state', type=int, default=[4, 4, 4], help='SSM state expansion factor')
    parser.add_argument('--en_d_conv', type=int, default=[3, 3, 3], help='local convolution width')
    parser.add_argument('--en_expand', type=int, default=[1, 1, 1], help='d_inner expansion factor')
    parser.add_argument('--en_dt_min', type=float, default=0.001, help='SSM dt_min')
    parser.add_argument('--en_dt_max', type=float, default=0.1, help='SSM dt_max')
    parser.add_argument('--en_dt_rank', type=str, default='auto', help='SSM dt_rank')
    parser.add_argument('--en_dt_init', type=str, default='random', help='SSM dt_init')
    parser.add_argument('--en_dt_scale', type=float, default=1.0, help='SSM dt_scale')
    parser.add_argument('--en_dt_init_floor', type=float, default=1e-4, help='SSM dt_init_floor')

    parser.add_argument('--en_conv_bias', type=bool, default=False,
                        help='whether to use bias in the convolution of SSM')
    parser.add_argument('--en_proj_bias', type=bool, default=False,
                        help='whether to use bias in the projection of SSM')
    parser.add_argument('--en_is_divide_out', type=bool, default=True,
                        help='whether to divide bidirectional SSM by 2 before the output')
    parser.add_argument('--en_is_out_norm', type=bool, default=False,
                        help='whether to use normalization after at the end of SSM')

    parser.add_argument('--en_bi_ssm', type=bool, default=True, help='whether to use bidirectional SSM')

    # for Mamba2
    parser.add_argument('--en_ngroups', type=int, default=1, help='number of groups in each layer')
    parser.add_argument('--en_headdim', type=int, default=64, help='head dimension in each layer')
    parser.add_argument('--en_chunk_size', type=int, default=256, help='chunk size')

    # --- decoder --- #
    parser.add_argument('--de_embed_dim', type=int, default=[192], help='embedding dimensions in the decoder model')
    parser.add_argument('--de_depths', type=int, default=[1], help='number of blocks inside each layer')
    parser.add_argument('--de_grouping_size', type=int, default=[(1, 472)],
                        help='grouping size for mamba/self-attention')
    parser.add_argument('--de_mlp_ratio', type=float, default=1., help='ratio of mlp hidden dim to embedding dim')
    parser.add_argument('--de_drop_rate', type=float, default=0., help='dropout rate')
    parser.add_argument('--de_drop_path_rate', type=float, default=0., help='stochastic depth rate')
    parser.add_argument('--de_embed_norm', type=bool, default=True,
                        help='if True, add normalization after point embedding')
    parser.add_argument('--de_use_checkpoint', type=bool, default=False,
                        help='whether to use checkpointing to save memory')
    parser.add_argument('--de_use_reentrant', type=bool, default=True,
                        help='whether to use reentrant for checkpoint')

    # decoder Mamba
    parser.add_argument('--de_d_state', type=int, default=[4], help='SSM state expansion factor')
    parser.add_argument('--de_d_conv', type=int, default=[3], help='local convolution width')
    parser.add_argument('--de_expand', type=int, default=[1], help='d_inner expansion factor')
    parser.add_argument('--de_dt_min', type=float, default=0.001, help='SSM dt_min')
    parser.add_argument('--de_dt_max', type=float, default=0.1, help='SSM dt_max')
    parser.add_argument('--de_dt_rank', type=str, default='auto', help='SSM dt_rank')
    parser.add_argument('--de_dt_init', type=str, default='random', help='SSM dt_init')
    parser.add_argument('--de_dt_scale', type=float, default=1.0, help='SSM dt_scale')
    parser.add_argument('--de_dt_init_floor', type=float, default=1e-4, help='SSM dt_init_floor')
    parser.add_argument('--de_conv_bias', type=bool, default=False,
                        help='whether to use bias in the convolution of SSM')
    parser.add_argument('--de_proj_bias', type=bool, default=False,
                        help='whether to use bias in the projection of SSM')
    parser.add_argument('--de_is_divide_out', type=bool, default=True,
                        help='whether to divide bidirectional SSM by 2 before the output')
    parser.add_argument('--de_is_out_norm', type=bool, default=False,
                        help='whether to use normalization after at the end of SSM')

    parser.add_argument('--de_bi_ssm', type=bool, default=True, help='whether to use bidirectional SSM')

    parser.add_argument('--de_ngroups', type=int, default=1, help='number of groups in each layer')
    parser.add_argument('--de_headdim', type=int, default=8, help='head dimension in each layer')
    parser.add_argument('--de_chunk_size', type=int, default=256, help='chunk size')

    # --- regression head ---
    parser.add_argument('--head_hidden_dim', type=int, default=64, help='hidden dimension for the regression head')
    parser.add_argument('--head_out_dim', type=int, default=7, help='output dimension for the regression head (delay in seconds)')
    parser.add_argument('--head_drop_rate', type=float, default=0., help='head dropout rate')
    parser.add_argument('--head_use_checkpoint', type=bool, default=False,
                        help='whether to use checkpointing to save memory')
    parser.add_argument('--head_use_reentrant', type=bool, default=True, help='whether to use reentrant for checkpoint')

    parser.add_argument('--pretrained_model', type=str, default=None,
                        help='pretrained model i.e. trained model with best loss')
    parser.add_argument('--training_checkpoint', type=str, default=None,
                        help='training checkpoint to resume training')

    # --- dataset class --- #
    parser.add_argument('--root_gtfs_logs', type=str,
                        default=r'./data/subwaydatanyc', help='root to the 1.8GB GTFS log CSV archive')
    parser.add_argument('--root_mta_static', type=str,
                        default=r'./data/mta_gtfs_static', help='root to MTA static metadata')

    parser.add_argument('--years_train', type=str, default=['2022', '2023'], help='years for training')
    parser.add_argument('--years_val', type=str, default=['2024'], help='years for validation')
    parser.add_argument('--years_test', type=str, default=['2024'], help='years for testing')

    parser.add_argument('--delta_t', type=int, default=4,
                        help='number of time steps in the historical context for the initial condition')
    parser.add_argument('--delta_t_f', type=int, default=7, help='forecast horizon lead time steps')

    parser.add_argument('--is_sample', type=bool, default=False, help='option to sample subset of stations')
    parser.add_argument('--n_points', type=int, default=472, help='total number of subway stations in the NYC graph')

    parser.add_argument('--graph_vertex_order', type=str, default='dfs',
                        help='algorithm to serialize graph vertices (dfs, bfs, shortest_path, etc.)')

    parser.add_argument('--nan_fill', type=float, default=0., help='a value to fill missing values')
    parser.add_argument('--is_shuffle', type=bool, default=False, help='if True, apply data samples shuffling')
    parser.add_argument('--is_norm', type=bool, default=True, help='if True, apply data normalization')

    # --- training parameters --- #
    parser.add_argument('--gpu_id', type=str, default="0", help='gpu ids: i.e. 0  (0,1,2, use -1 for CPU)')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--n_workers_train', type=int, default=8, help='number of workers for training multiprocessing')
    parser.add_argument('--n_workers_val', type=int, default=8, help='number of workers for validation multiprocessing')
    parser.add_argument('--pin_memory', type=bool, default=True,
                        help='allocate the loaded samples in GPU memory. Use it with training on GPU')

    parser.add_argument('--batch_size_train', type=int, default=16, help='batch size for training')
    parser.add_argument('--batch_size_val', type=int, default=16, help='batch size for validation')

    parser.add_argument('--n_epochs', type=int, default=100, help='number of epochs for training')
    parser.add_argument('--val_every_n_epochs', type=int, default=1, help='validate every n epochs')
    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer for training')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.000001, help='weight decay')
    parser.add_argument('--beta', type=float, default=0.9, help='beta momentum term for Adam/AdamW')
    parser.add_argument('--max_norm', type=float, default=10, help='maximum norm to clip gradient during training')

    parser.add_argument('--lr_scheduler', type=str, default='cosine', help='learning rate scheduler')
    parser.add_argument('--lr_warmup', type=int, default=1e-6, help='learning rate for warmup')
    parser.add_argument('--lr_warmup_epochs', type=int, default=0, help='number of epochs for warmup')
    parser.add_argument('--lr_min', type=float, default=1e-5, help='minimum learning rate')
    parser.add_argument('--lr_decay_step', type=int, default=20, help='learning rate step decay')
    parser.add_argument('--lr_decay_rate', type=float, default=0.9, help='learning rate decay')

    # --- input variables --- #
    parser.add_argument('--in_dynamic', type=int, default=2, help='number of dynamic variables in state maps')
    parser.add_argument('--in_static', type=int, default=5, help='number of static variables')

    parser.add_argument('--variables_dynamic', type=str,
                        default=[
                            'train_position_x',
                            'train_position_y',
                            'delay_s',
                            'occupancy_status'
                        ],
                        help='input dynamic variables from GTFS')

    parser.add_argument('--variables_static', type=str,
                        default=[
                            'station_lat',
                            'station_lon',
                            'track_count',
                            'transfers_count',
                            'average_daily_ridership'
                        ],
                        help='input static variables from MTA metadata')

    return parser


def read_arguments(train=True, print=True, save=True):
    parser = argparse.ArgumentParser()
    parser = add_all_arguments(parser)
    parser.add_argument('--phase', type=str, default='train')
    config = parser.parse_args()
    config.phase = 'train' if train else 'test'
    if print:
        print_options(config, parser)
    if save:
        save_options(config, parser)
    return config


def save_options(config, parser):

    if config.name is None or len(config.name) == 0:
        config.name = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))

    dir_log = os.path.join(config.dir_log, config.name)
    os.makedirs(dir_log, exist_ok=True)

    with open(dir_log + '/config.txt', 'wt') as config_file:
        message = ''
        message += '----------------- Options ---------------       -------------------\n\n'
        for k, v in sorted(vars(config).items()):
            if k in ['variables_dynamic', 'variables_static', 'years_train', 'years_val', 'years_test', 'dir_log',
                     'root_gtfs_logs', 'root_mta_static']:
                continue
            default = parser.get_default(k)
            comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<20}{}\n'.format(str(k), str(v), comment)

        comment = '\t[default: %s]' % str(parser.get_default('root_gtfs_logs'))
        message += '\n{:>25}: {:<20}{}\n'.format('root_gtfs_logs', vars(config)['root_gtfs_logs'], comment)

        comment = '\t[default: %s]' % str(parser.get_default('root_mta_static'))
        message += '{:>25}: {:<20}{}\n'.format('root_mta_static', vars(config)['root_mta_static'], comment)

        comment = '\t[default: %s]' % str(parser.get_default('dir_log'))
        message += '{:>25}: {:<20}{}\n'.format('dir_log', vars(config)['dir_log'], comment)

        message += '\n----------------- Input Variables -------      -------------------'
        message += '\n\n{:>20}: {}\n'.format('Variables Dynamic', str(config.variables_dynamic))
        message += '{:>20}: {}\n'.format('Variables Static', str(config.variables_static))

        message += '\n----------------- Years -----------------      -------------------'
        if config.phase == 'train':
            message += '\n\n{:>20}: {}'.format('Training', str(config.years_train))
            message += '\n{:>20}: {}\n'.format('Validation', str(config.years_val))
        else:
            message += '\n\n{:>20}: {}\n'.format('Testing', str(config.years_test))

        message += '\n----------------- End -------------------      -------------------'
        config_file.write(message)

    with open(dir_log + '/config.pkl', 'wb') as config_file:
        pickle.dump(config, config_file)


def print_options(config, parser):
    message = ''
    message += '----------------- Options ---------------       -------------------\n\n'
    for k, v in sorted(vars(config).items()):
        if k in ['variables_dynamic', 'variables_static', 'years_train', 'years_val', 'years_test', 'dir_log',
                 'root_gtfs_logs', 'root_mta_static']:
            continue
        default = parser.get_default(k)
        comment = '\t[default: %s]' % str(default)
        message += '{:>25}: {:<20}{}\n'.format(str(k), str(v), comment)

    comment = '\t[default: %s]' % str(parser.get_default('root_gtfs_logs'))
    message += '\n{:>25}: {:<20}{}\n'.format('root_gtfs_logs', vars(config)['root_gtfs_logs'], comment)

    comment = '\t[default: %s]' % str(parser.get_default('root_mta_static'))
    message += '{:>25}: {:<20}{}\n'.format('root_mta_static', vars(config)['root_mta_static'], comment)

    comment = '\t[default: %s]' % str(parser.get_default('dir_log'))
    message += '{:>25}: {:<20}{}\n'.format('dir_log', vars(config)['dir_log'], comment)

    message += '\n----------------- Input Variables -------      -------------------'
    message += '\n\n{:>20}: {}\n'.format('Variables Dynamic', str(config.variables_dynamic))
    message += '{:>20}: {}\n'.format('Variables Static', str(config.variables_static))

    message += '\n----------------- Years -----------------      -------------------'
    if config.phase == 'train':
        message += '\n\n{:>20}: {}'.format('Training', str(config.years_train))
        message += '\n{:>20}: {}\n'.format('Validation', str(config.years_val))
    else:
        message += '\n\n{:>20}: {}\n'.format('Testing', str(config.years_test))

    message += '\n----------------- End -------------------      -------------------'
    print(message)


if __name__ == '__main__':
    config = read_arguments(train=True, print=True, save=False)
