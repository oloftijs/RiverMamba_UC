
# ------------------------------------------------------------------
"""
Location-aware Adaptive Normalization layer https://arxiv.org/abs/2212.08208
Adapted for SubMamba


"""
# ------------------------------------------------------------------

import torch
import torch.nn as nn

# ------------------------------------------------------------------


class LOAN(nn.Module):
    """ Simple Location-aware Adaptive Normalization layer https://github.com/HakamShams/LOAN """

    def __init__(self, in_channels, cond_channels):
        super(LOAN, self).__init__()

        self.in_channels = in_channels
        self.cond_channels = cond_channels

        # projection layer
        self.mlp_beta = nn.Linear(cond_channels, in_channels, bias=True)
        self.act_mlp = nn.GELU()

        self.init_weights()

    def init_weights(self):

        def _init_weights(m):
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.normal_(m.weight.data, 0.0, 0.01)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias.data, 0.0)

            elif isinstance(m, nn.LayerNorm) and m.elementwise_affine:
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        self.apply(_init_weights)

    def forward(self, con_map: torch.Tensor):

        """ conditional map tensor con_map [*, K] """

        return self.act_mlp(self.mlp_beta(con_map))


if __name__ == '__main__':

    # Adjusted for NYC subway static features dimension (e.g. 10 static variables, 472 stations)
    x_static = torch.randn((1, 472, 10)).cuda()

    m = LOAN(in_channels=192, cond_channels=10).cuda()

    print(m)
    n_parameters = sum(p.numel() for p in m.parameters())
    print(f"number of parameters: {n_parameters}")

    print(m(x_static).shape)
