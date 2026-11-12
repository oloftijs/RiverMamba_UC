# ------------------------------------------------------------------
"""
Encoder SubMamba: Mamba2 backbone (Adapted from RiverMamba)


"""
# ------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint

try:
    from timm.layers import DropPath, trunc_normal_, lecun_normal_
except Exception:
    from timm.models.layers import DropPath, trunc_normal_, lecun_normal_

from functools import reduce, partial
from operator import mul
from mamba_ssm.modules.mamba2_simple import Mamba2Simple as Mamba_v2
from model.loan import LOAN
import math

try:
    import causal_conv1d  # noqa: F401
    HAS_CAUSAL_CONV1D = True
except Exception:
    HAS_CAUSAL_CONV1D = False

# - - - - - - - - - - - - - - - - - - - - - - - - - -

class MLP(nn.Module):
    """ Multilayer perceptron."""

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.,
                 use_checkpoint=False, use_reentrant=True):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

        self.use_checkpoint = use_checkpoint
        self.use_reentrant = use_reentrant

    def forward(self, x):
        if self.use_checkpoint:
            x = self.drop(self.act(checkpoint.checkpoint(self.fc1, x, use_reentrant=self.use_reentrant)))
            x = checkpoint.checkpoint(self.fc2, x, use_reentrant=self.use_reentrant)
        else:
            x = self.drop(self.act(self.fc1(x)))
            x = self.fc2(x)
        return x


def window_partition(x, grouping_size, order='spatial_first'):
    """
    Args:
        x: (B, T, P, C)
        grouping_size (tuple[int]): grouping_size
        order (string): 'temporal_first', 'spatial_first', 'temporal', 'spatial'
    Returns:
        windows: (B*num_groups, grouping_size[0]*grouping_size[1], C)
    """
    B, T, P, C = x.shape
    if order == 'temporal_first':
        x = x.view(B, T // grouping_size[0], grouping_size[0], P // grouping_size[1], grouping_size[1], C)
        windows = x.permute(0, 3, 1, 4, 2, 5).contiguous().view(-1, reduce(mul, grouping_size), C)
    elif order == 'spatial_first':
        x = x.view(B, T // grouping_size[0], grouping_size[0], P // grouping_size[1], grouping_size[1], C)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, reduce(mul, grouping_size), C)
    elif order == 'temporal':
        windows = x.permute(0, 2, 1, 3).contiguous().view(-1, T, C)
    elif order == 'spatial':
        windows = x.contiguous().contiguous().view(-1, P, C)
    return windows


def window_reverse(windows, grouping_size, B, T, P, order='spatial_first'):
    """
    Exact inverse of window_partition.
    Args:
        windows: (B*num_groups, grouping_size[0]*grouping_size[1], C)
        grouping_size (tuple[int]): grouping_size
        B (int): batch size
        T (int): temporal resolution
        P (int): number of the points
        order (string): 'temporal_first', 'spatial_first', 'temporal', 'spatial'
    Returns:
        x: (B, T, P, C)
    """
    if order == 'temporal_first':
        x = windows.view(B, P // grouping_size[1], T // grouping_size[0], grouping_size[1], grouping_size[0], -1)
        x = x.permute(0, 2, 4, 1, 3, 5).contiguous().view(B, T, P, -1)
    elif order == 'spatial_first':
        x = windows.view(B, T // grouping_size[0], P // grouping_size[1], grouping_size[0], grouping_size[1], -1)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, T, P, -1)
    elif order == 'temporal':
        x = windows.view(B, P, T, -1).permute(0, 2, 1, 3).contiguous()
    elif order == 'spatial':
        x = windows.view(B, T, P, -1).contiguous()
    return x


def apply_graph_order(x, graph_vertex_order):
    """
    Applies the topological graph vertex ordering for serialization.
    B, T, P, C = x.shape
    B, P = graph_vertex_order.shape
    """
    x_ordered = []
    for b in range(len(graph_vertex_order)):
        x_ordered.append(x[b, :, graph_vertex_order[b], :])
    return torch.stack(x_ordered, dim=0)


def reverse_graph_order(x, graph_vertex_order):
    """
    Reverses the topological ordering back to original graph nodes.
    """
    B, T, P, C = x.shape
    x_reversed = torch.zeros_like(x)
    for b in range(B):
        # Create inverse permutation indices
        inv_order = torch.argsort(graph_vertex_order[b])
        x_reversed[b] = x[b, :, inv_order, :]
    return x_reversed


class MambaBlock(nn.Module):
    """ SubMamba Encoder Block """
    def __init__(self,
                 dim,
                 dim_static,
                 grouping_size=(3, 10),
                 mlp_ratio=1.,
                 d_state=4,
                 expand=1,
                 d_conv=3,
                 dt_min=0.01,
                 dt_max=0.1,
                 dt_rank="auto",
                 dt_init="random",
                 dt_scale=1.0,
                 dt_init_floor=1e-4,
                 conv_bias=True,
                 proj_bias=False,
                 drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 curve_order='spatial_first',
                 bi_ssm=True,
                 use_checkpoint=False,
                 use_reentrant=True,
                 is_divide_out=True,
                 ngroups=1,
                 headdim=64,
                 chunk_size=256,
                 ):

        super().__init__()
        self.dim = dim
        self.dim_static = dim_static
        self.grouping_size = grouping_size

        self.mlp_ratio = mlp_ratio
        self.d_state = d_state
        self.expand = expand
        self.d_conv = d_conv
        self.dt_min = dt_min
        self.dt_max = dt_max
        self.dt_rank = dt_rank
        self.dt_init = dt_init
        self.dt_scale = dt_scale
        self.dt_init_floor = dt_init_floor
        self.conv_bias = conv_bias
        self.proj_bias = proj_bias

        self.curve_order = curve_order
        self.bi_ssm = bi_ssm

        self.use_checkpoint = use_checkpoint
        self.use_reentrant = use_reentrant

        self.is_divide_out = is_divide_out

        # The tensor has `dim` channels at both norms (Mamba2Simple widens internally).
        self.norm1 = norm_layer(dim, elementwise_affine=False)
        self.loan1 = LOAN(dim, dim_static)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False)
        self.loan2 = LOAN(dim, dim_static)

        # Mamba2Simple contains its own in_proj and out_proj: (B, L, dim) -> (B, L, dim)
        self.ssm = Mamba_v2(
            d_model=dim,
            d_state=d_state,
            d_conv=d_conv,
            conv_init=None,
            expand=expand,
            headdim=headdim,
            ngroups=ngroups,
            A_init_range=(1, 16),
            dt_min=dt_min,
            dt_max=dt_max,
            dt_init_floor=dt_init_floor,
            dt_limit=(0.0, float("inf")),
            learnable_init_states=False,
            activation="swish",
            bias=proj_bias,
            conv_bias=conv_bias,
            chunk_size=chunk_size,
            use_mem_eff_path=HAS_CAUSAL_CONV1D,
        )

        if bi_ssm:
            self.ssm_b = Mamba_v2(
                d_model=dim,
                d_state=d_state,
                d_conv=d_conv,
                conv_init=None,
                expand=expand,
                headdim=headdim,
                ngroups=1,
                A_init_range=(1, 16),
                dt_min=dt_min,
                dt_max=dt_max,
                dt_init_floor=dt_init_floor,
                dt_limit=(0.0, float("inf")),
                learnable_init_states=False,
                activation="swish",
                bias=proj_bias,
                conv_bias=conv_bias,
                chunk_size=chunk_size,
                use_mem_eff_path=HAS_CAUSAL_CONV1D,
            )

        self.mlp = MLP(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop,
                       use_checkpoint=False, use_reentrant=True)

        self.drop = nn.Dropout(drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward_part1(self, x, x_static):
        """ Forward part1 function """
        B, T, P, C = x.shape

        x = self.norm1(x)

        if self.use_checkpoint:
            x = x + checkpoint.checkpoint(self.loan1, x_static, use_reentrant=self.use_reentrant)
        else:
            x = x + self.loan1(x_static)

        pad_t = pad_p = 0
        pad_tt = (self.grouping_size[0] - T % self.grouping_size[0]) % self.grouping_size[0]
        pad_pp = (self.grouping_size[1] - P % self.grouping_size[1]) % self.grouping_size[1]
        x = F.pad(x, (0, 0, pad_p, pad_pp, pad_t, pad_tt))
        _, Tp, Pp, _ = x.shape

        if self.bi_ssm:
            x_b = window_partition(x.flip([-2]), self.grouping_size, order=self.curve_order)
        x = window_partition(x, self.grouping_size, order=self.curve_order)

        if self.use_checkpoint:
            x = checkpoint.checkpoint(self.ssm, x, use_reentrant=self.use_reentrant)
        else:
            x = self.ssm(x)

        if self.bi_ssm:
            if self.use_checkpoint:
                x_b = checkpoint.checkpoint(self.ssm_b, x_b, use_reentrant=self.use_reentrant)
            else:
                x_b = self.ssm_b(x_b)

        x = x.contiguous().view(-1, *(self.grouping_size + (C,)))
        x = window_reverse(x, self.grouping_size, B, Tp, Pp, order=self.curve_order)

        if self.bi_ssm:
            x_b = x_b.reshape(-1, *(self.grouping_size + (C,)))
            x_b = window_reverse(x_b, self.grouping_size, B, Tp, Pp, order=self.curve_order)

        if self.bi_ssm:
            x = (x + x_b.flip([-2])) / 2 if self.is_divide_out else x + x_b.flip([-2])

        if pad_tt > 0 or pad_pp > 0:
            x = x[:, :T, :P, :].contiguous()

        return x

    def forward_part2(self, x, x_static):
        """ Forward part2 function """
        x = self.norm2(x)
        if self.use_checkpoint:
            x = x + checkpoint.checkpoint(self.loan2, x_static, use_reentrant=self.use_reentrant)
        else:
            x = x + self.loan2(x_static)
        x = self.mlp(x)
        return x

    def forward(self, x, x_static, graph_vertex_order):
        """ Forward function """
        x = apply_graph_order(x, graph_vertex_order)
        x_static = apply_graph_order(x_static, graph_vertex_order)

        x = x + self.drop_path(self.forward_part1(x, x_static))
        x = x + self.drop_path(self.forward_part2(x, x_static))

        x = reverse_graph_order(x, graph_vertex_order)
        return x


class BasicLayer(nn.Module):
    """ A basic SubMamba layer """
    def __init__(self,
                 in_dim,
                 dim,
                 dim_static,
                 depth,
                 grouping_size=(3, 10),
                 mlp_ratio=4.,
                 d_state=4,
                 expand=1,
                 d_conv=3,
                 dt_min=0.01,
                 dt_max=0.1,
                 dt_rank="auto",
                 dt_init="random",
                 dt_scale=1.0,
                 dt_init_floor=1e-4,
                 conv_bias=True,
                 proj_bias=False,
                 drop=0.,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 embed_layer=None,
                 embed_norm=None,
                 curve_order='spatial_first',
                 bi_ssm=True,
                 is_divide_out=True,
                 use_checkpoint=False,
                 use_reentrant=True,
                 ngroups=1,
                 headdim=64,
                 chunk_size=256,
                 ):
        super().__init__()

        self.grouping_size = grouping_size
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.use_reentrant = use_reentrant
        self.dim = dim
        self.dim_static = dim_static
        self.in_dim = in_dim
        self.curve_order = curve_order
        self.bi_ssm = bi_ssm
        self.is_divide_out = is_divide_out

        if embed_layer:
            self.point_embed = embed_layer(in_chans=in_dim, embed_dim=dim, norm_layer=embed_norm)
        else:
            self.point_embed = None

        self.blocks = nn.ModuleList([
            MambaBlock(
                dim=dim,
                dim_static=dim_static,
                grouping_size=grouping_size,
                mlp_ratio=mlp_ratio,
                d_state=d_state,
                expand=expand,
                d_conv=d_conv,
                dt_min=dt_min,
                dt_max=dt_max,
                dt_rank=dt_rank,
                dt_init=dt_init,
                dt_scale=dt_scale,
                dt_init_floor=dt_init_floor,
                conv_bias=conv_bias,
                proj_bias=proj_bias,
                drop=drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                curve_order=curve_order,
                bi_ssm=bi_ssm,
                is_divide_out=is_divide_out,
                use_checkpoint=use_checkpoint,
                use_reentrant=self.use_reentrant,
                ngroups=ngroups,
                headdim=headdim,
                chunk_size=chunk_size
            )
            for i in range(depth)])

    def forward(self, x, x_static, graph_vertex_order):
        """ Forward function """
        if self.point_embed is not None:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(self.point_embed, x, use_reentrant=self.use_reentrant)
            else:
                x = self.point_embed(x)

        for blk in self.blocks:
            x = blk(x, x_static, graph_vertex_order)

        return x


class PointEmbed(nn.Module):
    """
    Embedding for the SSM layers
    Downsample the input by 2 along the T dimension
    """
    def __init__(self, in_chans=28, embed_dim=32, norm_layer=None):
        super().__init__()
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.proj = nn.Linear(in_chans * 2, embed_dim, bias=False)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim, elementwise_affine=False)
        else:
            self.norm = None

    def forward(self, x):
        B, T, P, C = x.shape
        pad_T = (T % 2 == 1) and T != 1

        if pad_T:
            x = F.pad(x, (0, 0, 0, 0, 0, T % 2))

        if T != 1:
            x0 = x[:, 0::2, :, :]
            x1 = x[:, 1::2, :, :]
        else:
            x0 = x[:, 0::2, :, :]
            x1 = x[:, 1::2, :, :]

        x = torch.cat([x0, x1], -1)
        x = self.proj(x)

        if self.norm is not None:
            x = self.norm(x)
        return x


class InputEmbed(nn.Module):
    """
    Embedding for the single subway dynamic input data
    """
    def __init__(self, in_chans=28, embed_dim=32):
        super().__init__()
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.embed = nn.Linear(in_chans, embed_dim, bias=True)
        self.act = nn.Tanh()

    def forward(self, x):
        return self.act(self.embed(x))


def _init_weights_m(
    module,
    n_layer,
    initializer_range=0.02,
    rescale_prenorm_residual=True,
    n_residuals_per_layer=2,
):
    if isinstance(module, nn.Linear):
        if module.bias is not None:
            if not getattr(module.bias, "_no_reinit", False):
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)

    if rescale_prenorm_residual:
        for name, p in module.named_parameters():
            if name in ["out_proj.weight", "fc2.weight"]:
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with torch.no_grad():
                    p /= math.sqrt(n_residuals_per_layer * n_layer)


class Mamba2(nn.Module):
    """
    Encoder SubMamba: Mamba2 backbone (Adapted from RiverMamba)
    """
    def __init__(self,
                 in_dynamic=4,
                 in_static=10,
                 curve_order='spatial_first',
                 bi_ssm=True,
                 embed_dynamic=32,
                 embed_norm=True,
                 embed_dim=None,
                 depths=None,
                 grouping_size=None,
                 mlp_ratio=1,
                 drop_rate=0.,
                 drop_path_rate=0.,
                 d_state=None,
                 expand=None,
                 d_conv=None,
                 dt_min=0.01,
                 dt_max=0.1,
                 dt_rank="auto",
                 dt_init="random",
                 dt_scale=1.0,
                 dt_init_floor=1e-4,
                 conv_bias=True,
                 proj_bias=False,
                 is_divide_out=True,
                 use_checkpoint=False,
                 use_reentrant=True,
                 ngroups=None,
                 headdim=None,
                 chunk_size=None,
                 ):

        super(Mamba2, self).__init__()

        self.in_dynamic = in_dynamic
        self.in_static = in_static

        self.curve_order = curve_order
        self.bi_ssm = bi_ssm
        self.is_divide_out = is_divide_out

        self.embed_dynamic = embed_dynamic
        self.embed_norm = embed_norm

        self.embed_dim = embed_dim if embed_dim is not None else [32, 32, 32]
        self.num_layers = len(self.embed_dim)
        self.depths = depths if depths is not None else [1 for _ in range(self.num_layers)]

        self.in_channels = embed_dynamic

        self.grouping_size = grouping_size if grouping_size is not None else [(4, 15), (2, 15), (1, 15)]
        self.d_state = d_state if d_state is not None else [16 for _ in range(self.num_layers)]
        self.expand = expand if expand is not None else [1 for _ in range(self.num_layers)]
        self.d_conv = d_conv if d_conv is not None else [3 for _ in range(self.num_layers)]

        self.ngroups = ngroups if ngroups is not None else [1 for _ in range(self.num_layers)]
        self.headdim = headdim if headdim is not None else [16 for _ in range(self.num_layers)]
        self.chunk_size = chunk_size if chunk_size is not None else [256 for _ in range(self.num_layers)]

        norm_layer = nn.LayerNorm

        self.mlp_ratio = mlp_ratio
        self.drop_rate = drop_rate
        self.drop_path_rate = drop_path_rate
        self.use_checkpoint = use_checkpoint
        self.use_reentrant = use_reentrant

        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, sum(self.depths))]

        self.layers = nn.ModuleList()

        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                in_dim=self.embed_dim[i_layer - 1] if i_layer > 0 else self.in_channels,
                dim=self.embed_dim[i_layer],
                dim_static=self.in_static,
                depth=self.depths[i_layer],
                grouping_size=self.grouping_size[i_layer],
                mlp_ratio=mlp_ratio,
                d_state=self.d_state[i_layer],
                expand=self.expand[i_layer],
                d_conv=self.d_conv[i_layer],
                dt_min=dt_min,
                dt_max=dt_max,
                dt_rank=dt_rank,
                dt_init=dt_init,
                dt_scale=dt_scale,
                dt_init_floor=dt_init_floor,
                conv_bias=conv_bias,
                proj_bias=proj_bias,
                drop=self.drop_rate,
                drop_path=dpr[sum(self.depths[:i_layer]):sum(self.depths[:i_layer + 1])],
                norm_layer=norm_layer,
                embed_layer=PointEmbed if i_layer > 0 else None,
                embed_norm=norm_layer if self.embed_norm else None,
                curve_order=curve_order,
                bi_ssm=bi_ssm,
                is_divide_out=is_divide_out,
                use_checkpoint=self.use_checkpoint,
                use_reentrant=self.use_reentrant,
                ngroups=self.ngroups[i_layer],
                headdim=self.headdim[i_layer],
                chunk_size=self.chunk_size[i_layer],
            )
            self.layers.append(layer)

        self.point_embed_dynamic = InputEmbed(in_dynamic, self.embed_dynamic)
        self.norm_input_embedding = nn.LayerNorm(self.in_channels, elementwise_affine=True)
        self.time_pos_embed = nn.Parameter(torch.zeros(1, 4, 1, self.in_channels))

        self.init_weights()

    def init_weights(self):
        def _init_weights(m):
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv3d):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0.)
                if isinstance(m, nn.Conv2d) and m.bias is not None:
                    nn.init.constant_(m.bias, 0.)
                if isinstance(m, nn.Conv3d) and m.bias is not None:
                    nn.init.constant_(m.bias, 0.)
            elif isinstance(m, nn.LayerNorm) and m.elementwise_affine:
                nn.init.constant_(m.bias, 0.)
                nn.init.constant_(m.weight, 1.0)

        self.point_embed_dynamic.apply(_init_weights)
        self.norm_input_embedding.apply(_init_weights)
        trunc_normal_(self.time_pos_embed, std=.02)

        for l in range(len(self.layers)):
            layer = self.layers[l]
            if layer.point_embed is not None:
                layer.point_embed.apply(_init_weights)

            for block in range(self.depths[l]):
                layer.blocks[block].apply(
                                        partial(
                                            _init_weights_m,
                                            n_layer=sum(self.depths),
                                        )
                                    )
                layer.blocks[block].loan1.init_weights()
                layer.blocks[block].loan2.init_weights()

    def forward(self, x_dynamic, x_static, graph_vertex_order):
        """
            Forward function
            Args:
                x_dynamic (torch.tensor): input state maps [B, T, P, V]
                x_static (torch.tensor): input static variables [B, P, V]
                graph_vertex_order (torch.tensor): input graph order indices [B, P]
            Returns:
                x (torch.tensor): output dynamic features [B, 1, P, K_embedding]
        """
        x_static = x_static[:, None, :, :]

        if self.use_checkpoint:
            x_dynamic = checkpoint.checkpoint(self.point_embed_dynamic, x_dynamic, use_reentrant=self.use_reentrant)
        else:
            x_dynamic = self.point_embed_dynamic(x_dynamic)

        # Apply positional embedding and normalisation to the single dynamic state input
        x = x_dynamic + self.time_pos_embed
        x = self.norm_input_embedding(x)

        for layer in self.layers:
            x = layer(x, x_static, graph_vertex_order)

        return x


if __name__ == '__main__':
    # Subway graph: ~470 stations (not the 82,804 river points of the original)
    P = 472
    data_dynamic = torch.randn((1, 4, P, 4)).cuda()
    data_static = torch.randn((1, P, 10)).cuda()
    graph_vertex_order = torch.arange(0, P).repeat(1, 1).cuda()

    model = Mamba2().cuda()
    print(model)
    n_parameters = sum(p.numel() for p in model.parameters())
    print(f"number of parameters: {n_parameters}")

    import time
    from torch import autocast

    with torch.no_grad():
        for i in range(10):
            time_start = time.time()
            with autocast(device_type='cuda', dtype=torch.bfloat16):
                test = model(data_dynamic, data_static, graph_vertex_order)
            print(time.time() - time_start)