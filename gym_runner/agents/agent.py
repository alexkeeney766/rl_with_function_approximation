from typing import Tuple, Union, Optional

import numpy as np
import torch

from gym_runner.q_func_approx import QualityFuncApprox


class Agent:
    """
    General class for a reinforcement learning agent to be used
    in an OpenAI gym environment. This class expects to use some 
    form of generalized q-function approximation e.g. linear function, 
    or neural network.
    """

    def __init__(
        self,
        q_func_approx: QualityFuncApprox,
        state_dim: Union[np.array, int, float, None] = None,
        num_actions: Optional[int] = None,
        gamma: float = 0.95,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.998,
    ) -> None:

        # Setting instance Vars
        self.q_func_approx = q_func_approx
        self.state_dim = state_dim

        self.num_actions = num_actions
        self.epsilon: float = epsilon
        self.epsilon_decay: float = epsilon_decay
        self.gamma: float = gamma
        self.eps_min: float = 0.01

    def choose_action(self, state: np.array) -> Tuple[int, torch.Tensor]:
        """
        Finds the maximum q-value over actions at a given state.
        returns the 'best' action and associated quality.
        """
        q = self.q_func_approx.forward(torch.tensor(state, dtype=torch.float32))
        action = int(q.argmax())

        return action, q

    def epsilon_greedy(self, state: np.array) -> Tuple[int, torch.Tensor]:
        """
        Randomly chooses to maximize expected reward or explore based
        on current value of epsilon parameter.
        """
        q = self.q_func_approx.forward(torch.tensor(state, dtype=torch.float32))
        if np.random.random() < self.epsilon:
            # Choose Random Action
            action: int = np.random.randint(self.num_actions)
        else:
            # Choose Greedy policy
            action = int(q.argmax())

        return action, q

    @staticmethod
    def copy_and_update(
        input: torch.Tensor, index: int, new_val: float
    ) -> torch.tensor:
        """
        Creates a copy of the 'input' tensor, with the value at 'index'
        replaced with 'new_value. Needed to generate the target given
        the Q-Function Approximation design.
        """
        output = input.detach().clone()
        output[index] = new_val
        return output

    def get_quality(self, state: np.array) -> torch.Tensor:
        """
        Get the quality vector over actions at the passed state.
        """
        return self.q_func_approx.forward(torch.tensor(state, dtype=torch.float32))

    def update_epsilon(self) -> None:
        """
        Update epsilon based on decay parameter.
        """
        self.epsilon = np.max([self.epsilon * self.epsilon_decay, self.eps_min])

    def init_training_episode(self, state: np.array) -> None:
        """
        This is run prior to each training episode: agent subclasses
        should independently implement this.
        """
        raise NotImplementedError

    def init_training_step(self) -> None:
        """
        This is run at the beggining of each training step: agent subclasses
        should independently implement this.
        """
        raise NotImplementedError

    def train_step(self, s_prime: np.array, reward: int, terminal: bool) -> None:
        """
        This is run after the action has been taken and s_prime, reward: agent subclasses
        should independently implement this.
        have been observed.
        """
        raise NotImplementedError

    def update(self, reward: float, q_prime: float, terminal: bool) -> None:
        """
        Internal function that should be called in train_step(),
        Not always neccessary
        """
        raise NotImplementedError

    def episode_aggregation_func(self) -> None:
        """
        This is run after each training episode.
        Ex: experience replay should be implemented here.
        Not always neccessary
        """
        raise NotImplementedError
