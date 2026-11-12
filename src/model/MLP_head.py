

# ------------------------------------------------------------------
"""
MLP regression head (Adapted from RiverMamba for SubMamba)


"""
# ------------------------------------------------------------------

import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_
import torch.utils.checkpoint as checkpoint

# ------------------------------------------------------------------


class MLP(nn.Module):
    def __init__(self, embed_dim=192, embed_dim_all=192*7, hidden_dim=64,
                 out_dim=1, drop_rate=0.1, use_checkpoint=False, use_reentrant=True):
        """
            Args:
                embed_dim (int, optional): input embedding dimension. Defaults to 192
                embed_dim_all (int, optional): input embedding dimension from other decoder layers. Defaults to 192*7
                hidden_dim (int, optional): hidden dimension. Defaults to 64
                out_dim (int, optional): output dimension. Defaults to 1 (delay in seconds).
                drop_rate (float, optional): dropout rate. Defaults to 0.1
                use_checkpoint (bool, optional): whether to use checkpoint. Defaults to False.
                use_reentrant (bool, optional): whether to use reentrants. Defaults to True.
        """

        super().__init__()

        self.embed_dim = embed_dim
        self.embed_dim_all = embed_dim_all
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim

        self.use_checkpoint = use_checkpoint
        self.use_reentrant = use_reentrant

        self.fc1 = nn.Linear(embed_dim, hidden_dim//2, bias=True)
        self.fc2 = nn.Linear(embed_dim_all, hidden_dim//2, bias=True)
        self.fc3 = nn.Linear(hidden_dim, out_dim, bias=True)
        self.act = nn.ReLU()

        self.dropout = nn.Dropout(drop_rate)
        self.dropout_f = nn.Dropout(0.4)

    def forward(self, x, x_future):
        """
            Forward function
            Args:
                x (torch.tensor): input features [B, 1, P, K]
                x_future (list of torch.tensor): input features [B, 1, P, K] * lead time
            Returns:
                x (torch.tensor): output [B, 1, P, out_dim]
        """

        # concatenate features from other decoder layers / lead time steps
        x_future = torch.cat(x_future, dim=-1)

        # dropout for features
        x = self.dropout(x)
        x_future = self.dropout_f(x_future)

        if self.use_checkpoint:
            x = checkpoint.checkpoint(self.fc1, x, use_reentrant=self.use_reentrant)
            x_future = checkpoint.checkpoint(self.fc2, x_future, use_reentrant=self.use_reentrant)
            x = self.act(torch.cat((x, x_future), dim=-1))
            x = checkpoint.checkpoint(self.fc3, x, use_reentrant=self.use_reentrant)
        else:
            x = self.fc1(x)
            x_future = self.fc2(x_future)
            x = self.act(torch.cat((x, x_future), dim=-1))
            x = self.fc3(x)

        return x


if __name__ == '__main__':
    """
    x      shape: extracted features from the corresponding decoder layer:  (B, 1, P, K)
    x_all  shape: extracted features from the other decoder layers:         (B, 1, P, K) * lead time (delta_t_f = 7)
    """

    x = torch.randn((2, 1, 472, 192), device='cuda')
    x_all = [torch.randn((2, 1, 472, 192), device='cuda')] * 7

    model = MLP(embed_dim=192, embed_dim_all=192*7, hidden_dim=64, out_dim=1, drop_rate=0.1).to('cuda')

    print(model)

    n_parameters = sum(p.numel() for p in model.parameters())
    print(f"number of parameters: {n_parameters}")

    test = model(x, x_all)
    print(test.shape)
