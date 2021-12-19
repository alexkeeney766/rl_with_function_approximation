from typing import List, Tuple, Union, Optional

import numpy as np
import torch

from q_func_approx import QualityFuncApprox
from .agent import Agent


class SarsaAgent(Agent):
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

    def init_training_episode(self, state: np.array) -> None:
        # Get Initial Action
        self.action, self.quality = self.choose_action(state)

    def init_training_step(self, state: np.array) -> None:
        pass

    def train_step(self, s_prime: np.array, reward: int) -> None:
        s_prime = s_prime
        reward = reward

        # Find max_a of new state S'
        a_prime, q_prime = self.epsilon_greedy(s_prime)

        # Update Weights
        self.update(self.quality, self.action, reward, q_prime)

        # Update current state and action
        state = s_prime
        self.action = a_prime
        self.quality = self.get_quality(state)

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

    def episode_aggregation_func(self) -> None:
        pass
