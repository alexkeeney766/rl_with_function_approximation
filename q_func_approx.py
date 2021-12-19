from typing import Callable

import torch.nn.functional as F
import torch


class QualityFuncApprox(torch.nn.Module):
    def __init__(
        self,
        num_actions: int,
        num_states: int,
        optimizer: str = "sgd",
        loss_func: str = "l1",
        alpha: float = 0.001,
    ):
        """
        Initialize a neural network that approximates a Q-function
        of a reinforcement learning agent. Should be used with SARSA,
        on policy agents.
        """
        super(QualityFuncApprox, self).__init__()

        # The layers of a network that maps from states to quality
        # values for each action given the current state.
        self.fc1 = torch.nn.Linear(num_states, 128)
        self.fc2 = torch.nn.Linear(128, num_actions)

        # Setting optimizer and loss function
        # TODO replace w/ enum
        self.optimizer: Callable = {
            "sgd": torch.optim.SGD(self.parameters(), lr=alpha),
            "adam": torch.optim.Adam(self.parameters(), lr=alpha),
        }[optimizer]
        self.loss_func: Callable = {
            "sum": lambda a, b: (b - a).sum(),
            "mse": F.mse_loss,
            "l1": F.smooth_l1_loss,
        }[loss_func]

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Computes the quality of each action given a current state s"""
        x = torch.tanh(self.fc1(state))
        return torch.relu(self.fc2(x))

    def update(self, x: torch.Tensor, target: torch.Tensor) -> None:
        """Updates the network given a """
        loss = self.loss_func(x, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
