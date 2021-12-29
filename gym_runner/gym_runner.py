from typing import Callable, Optional
import os

import gym
import numpy as np

from gym_runner.agents.sarsa_agent import SarsaAgent

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
        display_metrics: bool = False,
        max_steps: int = 200000,
        num_episodes: int = 1000,
    ):
        self.num_episodes: int = num_episodes
        self.max_steps = max_steps
        self.display_metrics = display_metrics
        self.env = gym.make(env_id)

    def train(
        self,
        agent: SarsaAgent,
        num_episodes: Optional[int] = None,
        plot_state: bool = False,
    ) -> np.array:
        """
        Train an agent on the current environment.
        """
        num_episodes = num_episodes or self.num_episodes
        accumulated_rewards = []

        for i, episode in enumerate(range(num_episodes)):
            # Init S
            state = self.env.reset()
            accumulated_reward = 0

            # Initialize agent-specific training
            agent.init_training_episode(state)

            # Loop for each step in episode:
            for _ in range(self.max_steps):

                # Agent Specific pre-step training process
                agent.init_training_step()

                if plot_state:
                    self.env.render()

                # Take Action A, observe reward R, next state S'
                s_prime, r, terminal, _ = self.env.step(agent.action)
                accumulated_reward += r

                # Agent Specific post-step training process
                agent.train_step(s_prime, r, terminal)

                if terminal:
                    accumulated_rewards.append(accumulated_reward)
                    agent.update_epsilon()
                    break

            # Agent Specific post-episode training process, E.G. Memory Replay
            agent.episode_aggregation_func()

            # Intermediate output
            if i % 10 == 0 and self.display_metrics:
                os.system("clear")
                clear_output()
                print("Epsilon: ", agent.epsilon)
                if len(accumulated_rewards) > 1:
                    print("Current Reward: ", accumulated_rewards[-1])
                print("Episode: ", episode)

        return accumulated_rewards

    def attempt(
        self, agent: SarsaAgent, num_episodes: int = 100, plot_state: bool = False
    ) -> np.array:
        """
        Attempt the environment using the passed agent.
        """
        accumulated_rewards = []

        for _ in range(num_episodes):

            # Init S
            state = self.env.reset()
            accumulated_reward = 0

            # Get initial action
            action, _ = agent.choose_action(state)

            # Loop for each step until terminal or maximum is reached
            for _ in range(self.max_steps):

                if plot_state:
                    self.env.render()

                # Take action A, observe R, S', adn meta-data
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

