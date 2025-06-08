import torch
import torch.nn as nn
import typing as tt
from torchrl.modules import NoisyLinear

class DoubleDQNModel(nn.Module):

    def __init__(self, input_shape: tt.Tuple[int, ...], n_actions: int):
        super(DoubleDQNModel, self).__init__()

        self.val_net = nn.Sequential(
            nn.Linear(input_shape, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

        self.noisy_layers = [
            NoisyLinear(input_shape, 1024), # 0
            NoisyLinear(1024, 1024),        # 1
            NoisyLinear(1024, 512),         # 2
            NoisyLinear(512, 512),          # 3
            NoisyLinear(512, n_actions)     # 4
        ]

        self.adv_net = nn.Sequential(
            self.noisy_layers[0],
            nn.ReLU(),
            self.noisy_layers[1],
            nn.ReLU(),
            self.noisy_layers[2],
            nn.ReLU(),
            self.noisy_layers[3],
            nn.ReLU(),
            self.noisy_layers[4]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        val_out = self.val_net(x)
        adv_out = self.adv_net(x)
        
        return val_out + (adv_out - adv_out.mean(dim=1, keepdim=True))

    def reset_noise(self):
        
        for n in self.noisy_layers:
            n.reset_noise()
