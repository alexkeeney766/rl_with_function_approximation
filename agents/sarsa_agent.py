from typing import Tuple, Union, Optional

import numpy as np
import torch

from q_func_approx import QualityFuncApprox


class SarsaAgent:
    def __init__(
        self,
        q_func_approx: QualityFuncApprox,
        num_states: Union[np.array, int, float, None] = None,
        num_actions: Optional[int] = None,
        gamma: float = 0.95,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.998,
    ) -> None:
        """
        Initialize a SARSA on policy reinforcement learning agent. 
        This class expects to use some form of generalized q-function
        approximation e.g. linear function, or neural network.
        """

        # Setting instance Vars
        self.q_func_approx = q_func_approx
        self.num_states = num_states

        self.num_actions = num_actions
        self.epsilon: float = epsilon
        self.epsilon_decay: float = epsilon_decay
        self.gamma: float = gamma
        self.eps_min: float = 0.1

    def choose_action(self, state: np.array) -> tuple:
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

    def update(
        self, q: torch.Tensor, action: int, reward: float, q_prime: float
    ) -> None:
        """
        Update the weights of the Q-Function Approximation.
        Since the network is designed to return all state-action
        values at a paticular state, We need the update target
        to be a copy of the current prediction, with only the 
        q-value of the observed state-action pair updated.
        """
        # We only want to update one
        updated_val = reward + self.gamma * q_prime.detach().clone()[action]
        target = self.copy_and_update(q, action, updated_val)

        self.q_func_approx.update(q, target)

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


class SarsaMemoryReplayAgent(SarsaAgent):
    pass
