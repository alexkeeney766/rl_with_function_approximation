from typing import Callable
from torch import optim

import torch.nn.functional as F
import torch


class QualityFuncApprox(torch.nn.Module):
    def __init__(
        self,
        num_actions: int,
        num_states: int,
        optimizer: str = "SGD",
        loss_func: str = "l1",
        alpha: float = 0.001,
    ):
        """
        Initialize a neural network that approximates a Q-function
        of a reinforcement learning agent.
        """
        super(QualityFuncApprox, self).__init__()
        self.model = self.make_model(num_states=num_states, num_actions=num_actions)

        # Setting optimizer and loss function
        # TODO replace w/ enum
        self.optimizer: Callable = {
            "SGD": torch.optim.SGD(self.model.parameters(), lr=alpha),
            "Adam": torch.optim.Adam(self.model.parameters(), lr=alpha),
        }[optimizer]

        self.loss_func: Callable = {
            "sum": lambda a, b: (b - a).sum(),
            "mse": F.mse_loss,
            "l1": F.smooth_l1_loss,
        }[loss_func]

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Computes the quality of each action given a current state s"""
        return self.model(state)

    def update(self, x: torch.Tensor, target: torch.Tensor) -> None:
        """Updates the network given a """
        loss = self.loss_func(x, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def make_model(self, num_actions: int, num_states: int,) -> torch.nn.Sequential:
        raise NotImplementedError


class QFuncLargeTwoLayer(QualityFuncApprox):
    def __init__(
        self,
        num_actions: int,
        num_states: int,
        optimizer: str = "SGD",
        loss_func: str = "l1",
        alpha: float = 0.001,
    ):
        """
        Initialize a neural network that approximates a Q-function
        of a reinforcement learning agent.
        """
        self.num_actions = num_actions
        self.num_states = num_states
        super().__init__(
            num_actions=num_actions,
            num_states=num_states,
            optimizer=optimizer,
            loss_func=loss_func,
            alpha=alpha,
        )

    def make_model(self, num_actions: int, num_states: int,) -> torch.nn.Sequential:
        """The layers of a network that maps from states to quality
        values for each action given the current state."""
        fc1 = torch.nn.Linear(num_states, 128)
        fc2 = torch.nn.Linear(128, num_actions)
        model = torch.nn.Sequential(fc1, torch.nn.Tanh(), fc2, torch.nn.ReLU())

        return model

    def __repr__(self):
        return f"{self.num_states} -> 128 -> {self.num_actions}"

class QFuncSmallThreelayer(QualityFuncApprox):
    def __init__(
        self,
        num_actions: int,
        num_states: int,
        optimizer: str = "SGD",
        loss_func: str = "l1",
        alpha: float = 0.001,
    ):
        """
        Initialize a neural network that approximates a Q-function
        of a reinforcement learning agent.
        """
        self.num_actions = num_actions
        self.num_states = num_states
        super().__init__(
            num_actions=num_actions,
            num_states=num_states,
            optimizer=optimizer,
            loss_func=loss_func,
            alpha=alpha,
        )

    def make_model(self, num_states, num_actions) -> torch.nn.Sequential:
        """The layers of a network that maps from states to quality
        values for each action given the current state."""
        fc1 = torch.nn.Linear(num_states, 12)
        fc2 = torch.nn.Linear(12, 12)
        fc3 = torch.nn.Linear(12, num_actions)
        model = torch.nn.Sequential(fc1, torch.nn.ReLU(), fc2, torch.nn.ReLU(), fc3)

        return model

    def __repr__(self):
        return f"{self.num_states} -> 12 -> 12 -> {self.num_actions}"