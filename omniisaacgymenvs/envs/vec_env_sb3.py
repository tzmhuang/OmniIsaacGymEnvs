from omni.isaac.gym.vec_env import VecEnvBase

import torch
import numpy as np

from datetime import datetime


# VecEnv Wrapper for RL training
class VecEnvSB3(VecEnvBase):

    def _process_data(self):
        self._obs = torch.clamp(self._obs, -self._task.clip_obs, self._task.clip_obs).to(self._task.rl_device).clone()
        self._rew = self._rew.to(self._task.rl_device).clone()
        self._states = torch.clamp(self._states, -self._task.clip_obs, self._task.clip_obs).to(self._task.rl_device).clone()
        self._resets = self._resets.to(self._task.rl_device).clone()
        self._extras = self._extras.copy()

    def set_task(
        self, task, backend="numpy", sim_params=None, init_sim=True
    ) -> None:
        super().set_task(task, backend, sim_params, init_sim)

        self.num_states = self._task.num_states
        self.state_space = self._task.state_space

    def getattr_depth_check(self, name, already_found):
        """Check if an attribute reference is being hidden in a recursive call to __getattr__

        :param name: name of attribute to check for
        :param already_found: whether this attribute has already been found in a wrapper
        :return: name of module whose attribute is being shadowed, if any.
        """
        if hasattr(self, name) and already_found:
            return f"{type(self).__module__}.{type(self).__name__}"
        else:
            return None

    def step(self, actions):
        if not isinstance(actions, torch.Tensor):
            actions = torch.Tensor(actions)
        actions = torch.clamp(actions, -self._task.clip_actions, self._task.clip_actions).to(self._task.device).clone()

        if self._task.randomize_actions:
            actions = self._task._dr_randomizer.apply_actions_randomization(actions=actions, reset_buf=self._task.reset_buf)

        self._task.pre_physics_step(actions)
        
        for _ in range(self._task.control_frequency_inv):
            self._world.step(render=self._render)
            self.sim_frame_count += 1

        self._obs, self._rew, self._resets, self._extras = self._task.post_physics_step()

        if self._task.randomize_observations:
            self._obs = self._task._dr_randomizer.apply_observations_randomization(observations=self._obs, reset_buf=self._task.reset_buf)

        self._states = self._task.get_states()
        self._process_data()
        
        # obs_dict = {"obs": self._obs, "states": self._states}

        return self._obs, self._rew, self._resets, self._extras

    def reset(self):
        """ Resets the task and applies default zero actions to recompute observations and states. """
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{now}] Running RL reset")

        self._task.reset()
        # NOTE: modify misc dim here
        misc_dim = 1
        actions = torch.zeros((self.num_envs, self._task.num_actions+misc_dim), device=self._task.device)
        obs, _, _, _ = self.step(actions)

        return obs
