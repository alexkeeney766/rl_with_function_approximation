import random
from collections import deque
from typing import Optional, Union

import numpy as np
import torch
from gym_runner.q_func_approx import QualityFuncApprox

from .agent import Agent


class QLearningAgent(Agent):
    """
    Algorithm:
    ----------
    Loop for each episode:
        Init State S

        Loop for each step in episode:
            Choose Action A based on S, using current policy w/ exploration
            Take Action A, observe reward R and next state S'
            Choose action A' based on S', using current greedy policy
            target <- R + gamma * Q(S', A')

            Update Q Function based on difference between target and current Q(S,A)
            S <- S'
    """

    def init_training_episode(self, state: np.array) -> None:
        self.state = state

    def init_training_step(self) -> None:
        self.action, self.q = self.epsilon_greedy(self.state)

    def train_step(self, s_prime: np.array, reward: int, terminal: bool) -> None:
        # Choose action A' based on S', using current greedy policy
        _, q_prime = self.choose_action(s_prime)

        # Update Weights
        self.update(reward, q_prime, terminal)

        # Update current state
        self.state = s_prime

    def update(self, reward: float, q_prime: float, terminal: bool) -> None:
        """
        Update the weights of the Q-Function Approximation.
        Since the network is designed to return all state-action
        values at a paticular state, We need the update target
        to be a copy of the current prediction, with only the 
        q-value of the observed state-action pair updated.
        """
        # Calculate target to be: R + gamma * Q(S', A')
        if terminal:
            updated_val = reward
        else:
            updated_val = reward + self.gamma * q_prime.detach().clone()[self.action]

        # We only want to update one Q value
        target = self.copy_and_update(self.q, self.action, updated_val)

        # Run backprop
        self.q_func_approx.update(self.q, target)

    def episode_aggregation_func(self) -> None:
        pass


class QLearningAgentExperienceReplay(Agent):
    """
    Algorithm:
    ----------
    Loop for each episode:
        Init State S

        Loop for each step in episode:
            Choose Action A based on S, using current policy w/ exploration
            Take Action A, observe reward R, next state S'
            Choose action A' based on S', using current greedy policy
            target <- R + gamma * Q(S', A')

            Store tuple of S, A, target in memory
            S <- S'
        
        Sample from Memory and update based on target, Q(S,A) pairs.
    """

    def __init__(
        self,
        q_func_approx: QualityFuncApprox,
        state_dim: Union[np.array, int, float, None] = None,
        num_actions: Optional[int] = None,
        gamma: float = 0.95,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.998,
        max_episode_len: int = 2000,
        batch_size: int = 8,
    ) -> None:

        super().__init__(
            q_func_approx=q_func_approx,
            state_dim=state_dim,
            num_actions=num_actions,
            gamma=gamma,
            epsilon=epsilon,
            epsilon_decay=epsilon_decay,
        )

        self.memory = deque(maxlen=max_episode_len)
        self.batch_size = batch_size

    def init_training_episode(self, state: np.array) -> None:
        self.state = state

    def init_training_step(self) -> None:
        self.action, self.q = self.epsilon_greedy(self.state)

    def train_step(self, s_prime: np.array, reward: int, terminal: bool) -> None:
        # Choose action A' based on S', using current greedy policy
        _, q_prime = self.choose_action(s_prime)

        # Update Weights
        self.update(reward, q_prime, terminal)

        # Update current state
        self.state = s_prime

    def update(self, reward: float, q_prime: float, terminal: bool) -> None:
        # Calculate target to be: R + gamma * Q(S', A')
        if terminal:
            target = reward
        else:
            target = reward + self.gamma * q_prime.detach().clone()[self.action]

        # Append step data to memory
        self.memory.append((self.state, self.action, target))

    def episode_aggregation_func(self) -> None:
        # Run experience replay on a sample of the episode's steps
        sample_exp = random.sample(self.memory, min(len(self.memory), self.batch_size))
        x_batch, y_batch = [], []

        for state, action, updated_val in sample_exp:
            q = self.get_quality(state)
            target = self.copy_and_update(q, action, updated_val)
            x_batch.append(q)
            y_batch.append(target)
        
        self.q_func_approx.update(torch.stack(x_batch) ,torch.stack(y_batch))

    