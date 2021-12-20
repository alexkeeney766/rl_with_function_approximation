from typing import Callable, Tuple, Optional
import pandas as pd
from sklearn.model_selection import ParameterGrid
import numpy as np
from multiprocessing.pool import Pool
from gym_runner import GymRunner
from q_func_approx import QualityFuncApprox
from agents.agent import Agent


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
        num_states = runner.env.observation_space.shape[0]

        if self.Q is not None:
            q: QualityFuncApprox = self.Q(
                num_actions=num_actions, num_states=num_states, **q_params
            )

        else:
            q: QualityFuncApprox = agent_params.pop("q_func_approx")(
                num_actions=num_actions, num_states=num_states, **q_params
            )

        agent: Agent = self.agent(
            q_func_approx=q,
            num_actions=num_actions,
            num_states=num_states,
            **agent_params
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
        return fold_rewards

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
        if num_procs > 1:
            with Pool(num_procs) as pool:
                rewards: np.array = pool.starmap(
                    self.k_repeats, self.results.iterrows()
                )
        else:
            rewards = []
            for i, row in self.results.iterrows():
                rewards.append(self.k_repeats(i, row))

        self.results["mean_score"] = -1
        self.results["max_mean_score"] = -1
        for i, fold_rewards in enumerate(rewards):
            self._results[i] = fold_rewards
            self.results.loc[i, "mean_score"] = fold_rewards.mean()
            self.results.loc[i, "max_mean_score"] = fold_rewards.mean(axis=1).max()

        return self.results

