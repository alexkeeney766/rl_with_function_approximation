from typing import Callable, Tuple
import gym
import pandas as pd
from sklearn.model_selection import ParameterGrid
import numpy as np
from multiprocessing.pool import Pool
from gym_runner import GymRunner
from q_func_approx import QualityFuncApprox
from agents.sarsa_agent import SarsaAgent


class GridSearch:
    def __init__(
        self,
        agent: Callable[[], SarsaAgent],
        Q: Callable[[],QualityFuncApprox],
        runner: Callable[[], GymRunner],
        param_grid: dict,
        env_id: str,
    ):
        self.Q = Q
        self.agent = agent
        self.env_id = env_id
        self.runner = runner
        self.param_grid = ParameterGrid(param_grid)
        self.results = pd.DataFrame.from_dict(self.param_grid, orient="columns")
        self._results = {}
        self.agent_param_keys = ["gamma", "epsilon", "epsilon_decay"]
        self.q_param_keys = ["optimizer", "loss_func", "alpha"]

    def run(self, agent_params: dict, q_params: dict):
        """
        Train and attempt a single agent and q-func instance given passed params
        """
        runner: GymRunner = self.runner(self.env_id)
        num_actions = runner.env.action_space.n
        num_states = runner.env.observation_space.shape[0]

        q: QualityFuncApprox = self.Q(
            num_actions=num_actions, num_states=num_states, **q_params
        )

        agent: SarsaAgent = self.agent(
            q, num_actions=num_actions, num_states=num_states, **agent_params
        )

        _ = runner.train(agent, num_episodes=1000)
        test_rewards: np.array = runner.attempt(agent, 100)

        return test_rewards

    def k_repeats(
        self, i: int, params:dict, k: int = 5
    ) -> np.array:
        '''Run K repeats of the same parameter selection'''
        agent_params, q_params = self.split_params(params)
        fold_rewards = np.array(
            [self.run(agent_params=agent_params, q_params=q_params) for _ in range(k)]
        )
        return fold_rewards

    def split_params(self, params: dict) -> Tuple[dict, dict]:
        '''Helper function to split parameters into Agent and Q-Function parameters'''
        agent_params = {
            param: val for param, val in params.items() if param in self.agent_param_keys
        }
        q_params = {
            param: val for param, val in params.items() if param in self.q_param_keys
        }
        return agent_params, q_params

    
    def fit(self, num_procs = 1):
        
        if num_procs > 1:
            with Pool(num_procs) as pool:
                rewards: np.array = pool.starmap(self.k_repeats, self.results.iterrows())
        else:
            rewards = []
            for i, row in self.results.iterrows():
                rewards.append(self.k_repeats(i, row))
        
        self.results["mean_score"] = -1
        self.results["max_mean_score"] = -1
        for i, fold_rewards in enumerate(rewards):
            self._results[i] = fold_rewards
            self.results.loc[i, 'mean_score'] = fold_rewards.mean()
            self.results.loc[i, 'max_mean_score'] = fold_rewards.mean(axis = 1).max()

        return self.results

