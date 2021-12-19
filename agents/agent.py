from typing import List, Tuple, Union, Optional

import numpy as np
import torch

from q_func_approx import QualityFuncApprox


class Agent:
    def __init__(
        self,
        q_func_approx: QualityFuncApprox,
        num_states: Union[np.array, int, float, None] = None,
        num_actions: Optional[int] = None,
        gamma: float = 0.95,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.998
    ) -> None:
        """
        Initialize a SARSA on policy reinforcement learning agent. 
        This class expects to use some form of generalized q-function
        approximation e.g. linear function, or neural network.

        Algorithm:
        Loop for each episode:
            Init State S
            Choose Action A based on S, using current policy w/ exploration

            Loop for each step in episode:
                Take Action A, observe reward R, next state S'
                Choose Action A' based on S' using current policy w/ exploration
                Calculate target to be: R + gamma * Q(S', A')
                
                Update Q Function based on difference between target and current Q(S,A)
                S, A <- S', A'

        """

        # Setting instance Vars
        self.q_func_approx = q_func_approx
        self.num_states = num_states

        self.num_actions = num_actions
        self.epsilon: float = epsilon
        self.epsilon_decay: float = epsilon_decay
        self.gamma: float = gamma
        self.eps_min: float = 0.1

    
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
        raise NotImplementedError

    def init_training_step(self, state: np.array) -> None:
        raise NotImplementedError

    def train_step(self, s_prime: np.array, reward: int) -> None:
        raise NotImplementedError

    def update(
        self, q: torch.Tensor, action: int, reward: float, q_prime: float
    ) -> None:
        raise NotImplementedError

    def episode_aggregation_func(self) -> None:
        raise NotImplementedError
