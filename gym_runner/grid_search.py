from multiprocessing.pool import Pool, IMapIterator, starmapstar
from typing import Callable, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import ParameterGrid

from gym_runner.agents.agent import Agent
from gym_runner.gym_runner import GymRunner
from gym_runner.q_func_approx import QualityFuncApprox


def istarmap(self, func, iterable, chunksize=1):
    """
    Solution for lazy map with multiple input arguments comes from: 
    https://stackoverflow.com/questions/57354700/starmap-combined-with-tqdm
    User Darkonaut
    """
    self._check_running()
    if chunksize < 1:
        raise ValueError(f"Chunksize must be 1+, not {chunksize}")

    task_batches = Pool._get_tasks(func, iterable, chunksize)
    result = IMapIterator(self)
    self._taskqueue.put(
        (
            self._guarded_task_generation(result._job, starmapstar, task_batches),
            result._set_length,
        )
    )
    return (item for chunk in result for item in chunk)


Pool.istarmap = istarmap


class GridSearch:
    """
    A class implementing grid search for an arbitrary agent and 
    environment across a range of hpyerparameters and Q function
    approximations.
    """

    def __init__(
        self,
        agent: Callable[[], Agent],
        runner: Callable[[], GymRunner],
        param_grid: dict,
        env_id: str,
        Q: Optional[Callable[[], QualityFuncApprox]] = None,
    ):

        self.agent = agent
        self.env_id = env_id
        self.runner = runner

        if "q_func_approx" in param_grid:
            self.Q = None
        elif Q is not None:
            self.Q = Q
        else:
            raise AttributeError(
                "Provide Q function approximator as argument or in Param Grid"
            )

        self.param_grid = ParameterGrid(param_grid)
        self.results = pd.DataFrame.from_dict(self.param_grid, orient="columns")
        self._results = {}
        self.agent_param_keys = ["q_func_approx", "gamma", "epsilon", "epsilon_decay"]
        self.q_param_keys = ["optimizer", "loss_func", "alpha"]

    def run(self, agent_params: pd.Series, q_params: pd.Series):
        """
        Train and attempt a single agent and q-func instance given passed
        params
        """

        runner: GymRunner = self.runner(self.env_id)
        num_actions = runner.env.action_space.n
        state_dim = runner.env.observation_space.shape[0]

        if self.Q is not None:
            q: QualityFuncApprox = self.Q(
                num_actions=num_actions, state_dim=state_dim, **q_params
            )

        else:
            q: QualityFuncApprox = agent_params.pop("q_func_approx")(
                num_actions=num_actions, state_dim=state_dim, **q_params
            )

        agent: Agent = self.agent(
            q_func_approx=q,
            num_actions=num_actions,
            state_dim=state_dim,
            **agent_params,
        )

        _ = runner.train(agent, num_episodes=1000)
        test_rewards: np.array = runner.attempt(agent, 100)

        return test_rewards

    def k_repeats(self, i: int, params: pd.Series, k: int = 5) -> np.array:
        """
        Run K repeats of the same parameter selection, reduces variance of
        the mean_score column.
        """
        agent_params, q_params = self.split_params(params)
        # Need to pass a copy of the params dictionary because it will be modified
        fold_rewards = np.array(
            [
                self.run(agent_params=agent_params.copy(), q_params=q_params)
                for _ in range(k)
            ]
        )
        return i, fold_rewards

    def split_params(self, params: dict) -> Tuple[dict, dict]:
        """Helper function to split parameters into Agent and Q-Function parameters"""

        agent_params = {
            param: val
            for param, val in params.items()
            if param in self.agent_param_keys
        }
        q_params = {
            param: val for param, val in params.items() if param in self.q_param_keys
        }
        return agent_params, q_params

    def fit(self, num_procs=1):
        """
        Search across all combinations of hyper parameter values passed.
        return a dataframe of testing scores and parameter values.
        """
        num_fits = self.results.shape[0]
        print(f"Running {num_fits} seperate agents through {self.env_id} 5 times each.")
        print(f"Using {num_procs} processes.")
        if num_procs > 1:
            with Pool(num_procs) as pool:
                # rewards: List[np.array] = pool.starmap(
                #     self.k_repeats, self.results.iterrows()
                # )
                rewards: List[Tuple[int, np.array]] = []
                for reward in tqdm(
                    pool.istarmap(
                        func=self.k_repeats, iterable=self.results.iterrows()
                    ),
                    total=num_fits,
                ):
                    rewards.append(reward)
        else:
            rewards = []
            for i, row in tqdm(self.results.iterrows(), total=num_fits):
                rewards.append((i, self.k_repeats(i, row)))

        self.results["mean_score"] = -1
        self.results["max_mean_score"] = -1
        for i, fold_rewards in rewards:
            self._results[i] = fold_rewards
            self.results.loc[i, "mean_score"] = fold_rewards.mean()
            self.results.loc[i, "max_mean_score"] = fold_rewards.mean(axis=1).max()

        return self.results

