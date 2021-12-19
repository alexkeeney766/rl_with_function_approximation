from typing import Callable
import os

import gym
import numpy as np

from agents.sarsa_agent import SarsaAgent

try:
    from IPython.display import clear_output
except ModuleNotFoundError:

    def clear_output():
        None


class GymRunner:
    """
    Contains all information and functionality to run an arbitrary agent
    in an arbirary Open AI gym environment
    """

    def __init__(
        self,
        env_id: str,
        display: bool = False,
        max_steps: int = 200000,
        num_episodes: int = 1000,
    ):
        self.num_episodes: int = num_episodes
        self.max_steps = max_steps
        self.display = display
        self.env = gym.make(env_id)

    def train(self, agent: SarsaAgent, num_episodes: int, plot_func=None) -> np.array:
        '''
        Train an agent on the current environment.
        '''
        num_episodes = num_episodes or self.num_episodes
        accumulated_rewards = []
        
        for i, episode in enumerate(range(num_episodes)):
            # Init S and clear gradients from pervious step
            state = self.env.reset()
            accumulated_reward = 0

            # Get initial Action
            action, quality = agent.choose_action(state)

            # Loop for each step of unknown length
            for _ in range(self.max_steps):

                if plot_func is not None:
                    plot_func()

                # Take Action A, observe next state, reward and meta-data
                s_prime, r, terminal, _ = self.env.step(action)
                accumulated_reward += r

                if terminal:
                    accumulated_rewards.append(accumulated_reward)
                    agent.update_epsilon()
                    break

                # Find max_a of new state S'
                a_prime, q_prime = agent.epsilon_greedy(s_prime)

                # Update Weights
                agent.update(quality, action, r, q_prime)

                # Update current state and action
                state = s_prime
                action = a_prime
                quality = agent.get_quality(state)

            # Intermediate output
            if i % 10 == 0 and self.display:
                os.system("clear")
                clear_output()
                print("Epsilon: ", agent.epsilon)
                if len(accumulated_rewards) > 1:
                    print("Current Reward: ", accumulated_rewards[-1])
                print("Episode: ", episode)
        
        return accumulated_rewards
        
    def attempt(self, agent: SarsaAgent, num_episodes: int, plot_func=None) -> np.array:
        '''
        Attempt the environment using the passed agent.
        '''
        
        num_episodes = num_episodes or self.num_episodes
        accumulated_rewards = []

        for _ in range(num_episodes):

            # Init S
            state = self.env.reset()
            accumulated_reward = 0

            # Get initial action
            action, _ = agent.choose_action(state)

            # Loop for each step until terminal or maximum is reached
            for _ in range(self.max_steps):
                
                if plot_func is not None:
                    plot_func()
                
                ## Take action A, observe R, S', adn meta-data
                s_prime, r, terminal, _ = self.env.step(action)
                accumulated_reward += r

                # Check if terminal state is reached
                if terminal:
                    accumulated_rewards.append(accumulated_reward)
                    break

                # Find next best action at S'
                a_prime, _ = agent.choose_action(s_prime)

                # Update current state and action
                state = s_prime
                action = a_prime
            
            return np.array(accumulated_rewards)
                