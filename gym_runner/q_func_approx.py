from typing import Callable, List

import torch.nn.functional as F
import torch


class QualityFuncApprox(torch.nn.Module):
    def __init__(
        self,
        num_actions: int,
        state_dim: int,
        optimizer: str = "SGD",
        loss_func: str = "l1",
        alpha: float = 0.001,
    ):
        """
        Initialize a neural network that approximates a Q-function
        of a reinforcement learning agent.
        """
        super(QualityFuncApprox, self).__init__()
        self.model = self.make_model(state_dim=state_dim, num_actions=num_actions)

        # Setting optimizer and loss function
        # TODO replace w/ enum
        self.optimizer: Callable = {
            "SGD": torch.optim.SGD(self.model.parameters(), lr=alpha),
            "Adam": torch.optim.Adam(
                self.model.parameters(), lr=alpha, weight_decay=0.01
            ),
        }[optimizer]

        self.loss_func: Callable = {
            "sum": lambda a, b: (b - a).sum(),
            "mse": F.mse_loss,
            "l1": F.smooth_l1_loss,
        }[loss_func]

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Computes the quality of each action given a current state s"""
        return self.model(state)

    def update(self, y: torch.Tensor, target: torch.Tensor) -> None:
        """Updates the network given a """
        loss = self.loss_func(y, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def make_model(self, num_actions: int, state_dim: int,) -> torch.nn.Sequential:
        raise NotImplementedError


class QFuncLargeTwoLayer(QualityFuncApprox):
    def __init__(
        self,
        num_actions: int,
        state_dim: int,
        optimizer: str = "SGD",
        loss_func: str = "l1",
        alpha: float = 0.001,
    ):
        """
        Initialize a neural network that approximates a Q-function
        of a reinforcement learning agent.
        """
        self.num_actions = num_actions
        self.state_dim = state_dim
        super().__init__(
            num_actions=num_actions,
            state_dim=state_dim,
            optimizer=optimizer,
            loss_func=loss_func,
            alpha=alpha,
        )

    def make_model(self, num_actions: int, state_dim: int,) -> torch.nn.Sequential:
        """The layers of a network that maps from states to quality
        values for each action given the current state."""
        fc1 = torch.nn.Linear(state_dim, 128)
        fc2 = torch.nn.Linear(128, num_actions)
        model = torch.nn.Sequential(fc1, torch.nn.Tanh(), fc2, torch.nn.ReLU())

        return model

    def __repr__(self):
        return f"{self.state_dim} -> 128 -> {self.num_actions}"

    def __str__(self):
        return f"{self.state_dim} -> 128 -> {self.num_actions}"


class QFuncSmallThreelayer(QualityFuncApprox):
    def __init__(
        self,
        num_actions: int,
        state_dim: int,
        optimizer: str = "SGD",
        loss_func: str = "l1",
        alpha: float = 0.001,
    ):
        """
        Initialize a neural network that approximates a Q-function
        of a reinforcement learning agent.
        """
        self.num_actions = num_actions
        self.state_dim = state_dim
        super().__init__(
            num_actions=num_actions,
            state_dim=state_dim,
            optimizer=optimizer,
            loss_func=loss_func,
            alpha=alpha,
        )

    def make_model(self, state_dim, num_actions) -> torch.nn.Sequential:
        """The layers of a network that maps from states to quality
        values for each action given the current state."""
        fc1 = torch.nn.Linear(state_dim, 12)
        fc2 = torch.nn.Linear(12, 12)
        fc3 = torch.nn.Linear(12, num_actions)
        model = torch.nn.Sequential(fc1, torch.nn.ReLU(), fc2, torch.nn.ReLU(), fc3)

        return model

    def __repr__(self):
        return f"{self.state_dim} -> 12 -> 12 -> {self.num_actions}"

    def __str__(self):
        return f"{self.state_dim} -> 12 -> 12 -> {self.num_actions}"


class QFuncMedThreelayer(QualityFuncApprox):
    def __init__(
        self,
        num_actions: int,
        state_dim: int,
        optimizer: str = "SGD",
        loss_func: str = "l1",
        alpha: float = 0.001,
    ):
        """
        Initialize a neural network that approximates a Q-function
        of a reinforcement learning agent.
        """
        self.num_actions = num_actions
        self.state_dim = state_dim
        super().__init__(
            num_actions=num_actions,
            state_dim=state_dim,
            optimizer=optimizer,
            loss_func=loss_func,
            alpha=alpha,
        )

    def make_model(self, state_dim, num_actions) -> torch.nn.Sequential:
        """The layers of a network that maps from states to quality
        values for each action given the current state."""
        model = torch.nn.Sequential(
            torch.nn.Linear(state_dim, 24),
            torch.nn.Tanh(),
            torch.nn.Linear(24, 48),
            torch.nn.Tanh(),
            torch.nn.Linear(48, num_actions),
        )

        return model

    def __repr__(self):
        return f"{self.state_dim} -> 24 -> 48 -> {self.num_actions}"

    def __str__(self):
        return f"{self.state_dim} -> 24 -> 48 -> {self.num_actions}"


class QFuncFree(torch.nn.Module):
    def __init__(
        self,
        num_actions: int,
        state_dim: int,
        model_layers: List[torch.nn.Module],
        optimizer: str = "SGD",
        loss_func: str = "l1",
        alpha: float = 0.001,
    ):
        """
        Initialize a neural network that approximates a Q-function
        of a reinforcement learning agent.
        """
        super(QFuncFree, self).__init__()

        self.num_actions = num_actions
        self.state_dim = state_dim
        self.model_layers = model_layers
        self.model = self.make_model(num_actions=num_actions, state_dim=state_dim)

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

    def make_model(self, state_dim, num_actions) -> torch.nn.Sequential:
        return torch.nn.Sequential(*self.model_layers)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Computes the quality of each action given a current state s"""
        return self.model(state)

    def update(self, x: torch.Tensor, target: torch.Tensor) -> None:
        """Updates the network given a """
        loss = self.loss_func(x, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
