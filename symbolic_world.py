import numpy as np
import sympy as sp
import signal
import torch
import gymnasium as gym
from gymnasium import spaces


from bfgs import bfgs
from utils import seq_to_tree, handle_timeout
from expression import TargetExpression, AgentExpression


class SymbolicWorldEnv(gym.Env):
    def __init__(self, cfg, cal_r2=True):
        self._max_step = cfg["max_step"]  # The maximal number of steps in an episode
        self._cal_r2 = cal_r2
        if cal_r2:
            signal.signal(signal.SIGALRM, handle_timeout)

        """
        The following dictionary maps abstract actions from `self.action_space` to
        the operation.
        I.e. 0 corresponds to x_1, etc.
        """
        self._action_to_operation = {
            0: "x_1",
            1: "x_2",
            2: "c",
            3: "Abs",
            4: "+",
            5: "*",
            6: "/",
            7: "sqrt",
            8: "exp",
            9: "log",
            10: "**",
            11: "sin",
            12: "cos",
            13: "tan",
            14: "asin",
            15: "acos",
            16: "atan",
            17: "sinh",
            18: "cosh",
            19: "tanh",
            20: "coth",
            21: "-3",
            22: "-2",
            23: "-1",
            24: "0",
            25: "1",
            26: "2",
            27: "3",
            28: "4",
            29: "5",
        }

        # Observations
        self.observation_space = spaces.Dict(
            {
                "point_set": spaces.Box(
                    low=-np.Inf, high=np.Inf, shape=(1, 3, 100), dtype=np.float32
                ),
                "tree": spaces.MultiBinary(
                    [self._max_step, len(self._action_to_operation)]
                ),
            }
        )
        self.action_space = spaces.Discrete(len(self._action_to_operation))

        self._target_expr = None  # The target expression
        self._agent_expr = None  # The agent's expression

        self._step = 0

        self._device = None

    def get_obs(self):
        return {
            "point_set": self._target_expr.point_set.transpose(1, 2),
            "tree": seq_to_tree(self._agent_expr.opt_sequence, self._max_step)
            .unsqueeze(0)
            .to(self._device),
        }

    def get_info(self):
        return {
            "agent_expr": self._agent_expr,
            "target_expr": self._target_expr,
            "terminated": None,
            "R2": None,
        }

    def _cal_R2(self):
        point_set = self._target_expr.point_set
        x = point_set[0, :, :2].cpu()
        y = point_set[0, :, -1].cpu()

        try:
            signal.alarm(20)
            candidate_expr, _, mse, _, _ = bfgs(
                self._agent_expr.skeleton, x.unsqueeze(0), y
            )
        except Exception as e:
            print(f"Encountered in bfgs: {e}")
            mse = np.nan
        finally:
            signal.alarm(0)

        if np.isnan(mse) or np.isinf(mse):
            return 0

        self._agent_expr.expr = candidate_expr

        agent_expr = str(candidate_expr)

        try:
            signal.alarm(20)
            total_variables = ["x_1", "x_2"]
            X_dict = {x_: x[:, idx].cpu() for idx, x_ in enumerate(total_variables)}
            y_pred = sp.lambdify(",".join(total_variables), sp.sympify(agent_expr))(
                **X_dict
            )

            r2 = (
                1
                - torch.sum(torch.square(y - y_pred))
                / torch.sum(torch.square(y - torch.mean(y)))
            ).item()
        except Exception as e:
            r2 = 0
        finally:
            signal.alarm(0)

        return r2

    def reset(self, point_set, expr, skeleton=None, opt_sequence=None, seed=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # device
        self._device = point_set.device

        # Initialize the step counter
        self._step = 0

        # target expression
        self._target_expr = TargetExpression(point_set, expr, skeleton, opt_sequence)
        # agent expression
        self._agent_expr = AgentExpression(point_set)

        observation = self.get_obs()
        info = self.get_info()

        return observation, info

    def step(self, action):
        # Map the action (element of {0,1,2,3,...}) to the operation
        operation = self._action_to_operation[action]

        # Apply the operation to the agent expression
        all_replaced = self._agent_expr.add_opt(operation)

        # Increment the step counter
        self._step += 1

        observation = self.get_obs()
        info = self.get_info()
        reward = 0
        terminated = False

        if all_replaced:
            terminated = True
            info["terminated"] = "all_replaced"

            if self._cal_r2:
                try:
                    r2 = self._cal_R2()
                    reward = max(0, r2)
                except Exception as e:
                    reward = 0
                    print(f"Exception encountered in env: {e}")

        elif self._step == self._max_step:
            terminated = True
            info["terminated"] = "time_out"
            reward = 0

        return (
            observation,
            torch.tensor([reward]).to(self._device),
            terminated,
            False,
            info,
        )


if __name__ == "__main__":
    pass
