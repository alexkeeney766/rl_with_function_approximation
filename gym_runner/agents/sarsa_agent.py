import numpy as np

from .agent import Agent


class SarsaAgent(Agent):
    """
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
    
    def init_training_episode(self, state: np.array) -> None:
        # Get Initial Action
        self.action, self.q = self.choose_action(state)

    def init_training_step(self) -> None:
        pass

    def train_step(self, s_prime: np.array, reward: int, terminal:bool) -> None:

        # Choose Action A' based on S' using current policy w/ exploration
        a_prime, q_prime = self.epsilon_greedy(s_prime)

        # Update Weights
        self.update(reward, q_prime, terminal)

        # Update current state and action
        state = s_prime
        self.action = a_prime

        # Need to recompute avoid gradient issues after updateing
        self.q = self.get_quality(state)

    def update(self, reward: float, q_prime: float, terminal:bool) -> None:
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
