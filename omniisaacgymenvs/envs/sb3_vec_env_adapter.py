
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Type, Union

import numpy as np
import torch as th
from stable_baselines3.common.vec_env import VecEnvWrapper

from omni.isaac.gym.vec_env import VecEnvBase

# Define type aliases here to avoid circular import
# Used when we want to access one or more VecEnv
VecEnvIndices = Union[None, int, Iterable[int]]
# VecEnvObs is what is returned by the reset() method
# it contains the observation for each env
VecEnvObs = Union[th.Tensor, Dict[str, th.Tensor], Tuple[th.Tensor, ...]]
# VecEnvStepReturn is what is returned by the step() method
# it contains the observation, reward, done, info for each env
VecEnvStepReturn = Tuple[VecEnvObs, th.Tensor, th.Tensor, List[Dict]]

class VecAdapter(VecEnvWrapper):
    """
    Convert VecEnvBase object to a Stable-Baselines3 (SB3) VecEnv.
    :param venv: The VecEnvBase object.
    """

    def __init__(self, venv: VecEnvBase, reward_shaper=None):
        # Retrieve the number of environments from the config
        super().__init__(venv=venv)
        self.reward_shaper=reward_shaper # reward shapped in policy_algorithm class

    def step_async(self, actions: th.Tensor) -> None:
        self.actions = actions

    def reset(self) -> VecEnvObs:
        obs = self.venv.reset()
        return obs.cpu().numpy() # convert to numpy array for SB3
        # return obs

    def step_wait(self) -> VecEnvStepReturn:

        obs, rewards, resets, info_dict = self.venv.step(self.actions)
        # terminal ovservation
        #   -TODO


        # Convert extra (dict of arrays) to extra (list of dicts)

        infos = []
        # Convert dict of array to list of dict
        # and add terminal observation
        for i in range(self.num_envs):
            infos.append(
                {
                    key: info_dict[key][i]
                    for key in info_dict.keys()
                    if isinstance(info_dict[key], th.Tensor)
                }
            )
            if resets[i]:
                infos[i]["terminal_observation"] = obs[i].cpu().numpy()
            #     obs[i] = self.venv.reset(np.array([i]))[0]

        # convert to numpy array for SB3
        return obs.cpu().numpy(), rewards.cpu().numpy(), resets.cpu().numpy().astype(np.bool), infos
        # return obs, rewards, resets, infos