import torch
import torch.nn as nn
import typing as tt
from torchrl.modules import NoisyLinear

class DoubleDQNModel(nn.Module):
    """
    A Double Deep Q-Network (DQN) model with a dueling architecture and noisy layers.

    This model separates the Q-value estimation into a state value stream and an
    advantage stream. Noisy linear layers are used in the advantage stream for
    exploration, replacing traditional epsilon-greedy exploration.
    """

    def __init__(self, input_shape: tt.Tuple[int, ...], n_actions: int):
        """
        Initializes the DoubleDQNModel.

        Args:
            input_shape: The shape of the input state observations.
            n_actions: The number of possible actions in the environment.
        """
        super(DoubleDQNModel, self).__init__()

        self.val_net = nn.Sequential(
            nn.Linear(input_shape, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

        self.noisy_layers = [
            NoisyLinear(input_shape, 512), # 0
            NoisyLinear(512, 512),         # 1
            NoisyLinear(512, n_actions)    # 2
        ]

        self.adv_net = nn.Sequential(
            self.noisy_layers[0],
            nn.ReLU(),
            self.noisy_layers[1],
            nn.ReLU(),
            self.noisy_layers[2]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass of the network.

        Args:
            x: The input tensor representing the current state.

        Returns:
            The Q-values for each action.
        """

        val_out = self.val_net(x)
        adv_out = self.adv_net(x)
        
        return val_out + (adv_out - adv_out.mean(dim=1, keepdim=True))

    def reset_noise(self):
        """
        Resets the noise in all NoisyLinear layers.

        This should be called after each training step or episode to sample new noise
        for exploration.
        """
        
        for n in self.noisy_layers:
            n.reset_noise()
