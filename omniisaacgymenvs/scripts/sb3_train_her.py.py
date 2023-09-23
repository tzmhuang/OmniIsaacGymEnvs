# Copyright (c) 2018-2022, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


from omniisaacgymenvs.utils.hydra_cfg.hydra_utils import *
from omniisaacgymenvs.utils.hydra_cfg.reformat import omegaconf_to_dict, print_dict
from omniisaacgymenvs.utils.rlgames.rlgames_utils import RLGPUAlgoObserver, RLGPUEnv
from omniisaacgymenvs.utils.task_util import initialize_task
from omniisaacgymenvs.utils.config_utils.path_utils import retrieve_checkpoint_path
from omniisaacgymenvs.envs.vec_env_sb3 import VecEnvSB3
from omniisaacgymenvs.envs.sb3_vec_env_adapter import VecAdapter

import hydra
from omegaconf import DictConfig

from stable_baselines3 import HER, DDPG
from stable_baselines3.common.logger import configure
# from stable_baselines3.common.vec_env import VecMonitorGPU
from stable_baselines3.common.vec_env import VecMonitor, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.utils import safe_mean

from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Type, Union

# from rl_games.common import env_configurations, vecenv
# from rl_games.torch_runner import Runner

import datetime
import os
import torch
import numpy as np

class CallbackAdaptiveSchedulePPO(BaseCallback):
    """Learning rate = Initial learning rate * training/std"""
    
    def __init__(self, kl_threshold=0.008):
        super().__init__()
        self._kl_threshold = 0.008
        self._min_lr = 1e-6
        self._max_lr = 1e-2

    def _on_rollout_end(self) -> None:
        current_lr = self.model.learning_rate
        if kl_dist > (2.0 * self._kl_threshold):
            lr = max(current_lr / 1.5, self._min_lr)
        if kl_dist < (0.5 * self.kl_threshold):
            lr = min(current_lr * 1.5, self._max_lr)
        self.model.learning_rate = lr
        self.model._setup_lr_schedule()


class CallbackHyperparamsSchedulePPO(BaseCallback):
    """Learning rate = Initial learning rate * training/std"""
    
    def __init__(self):
        super().__init__()
        self._learning_rate_start = None
        self._kl_threshold = 0.008
        self._min_lr = 1e-6
        self._max_lr = 1e-2

    def _on_step(self):
        return True

    # def _on_rollout_end(self) -> None:
    #     if self._learning_rate_start is None:
    #         self._learning_rate_start = self.model.learning_rate
    #     # std = torch.exp(self.model.policy.log_std).mean().item()
    #     self.model.learning_rate = 0.1*self._learning_rate_start * self.model._current_progress_remaining
    #     self.model._setup_lr_schedule()

    def _on_rollout_end(self) -> None:
        # schedule lr
        kl_dist = self.model.approx_kl
        if kl_dist is not None:
            lr = current_lr = self.model.learning_rate
            if kl_dist > (2.0 * self._kl_threshold):
                print("up")
                lr = max(current_lr / 1.5, self._min_lr)
            if kl_dist < (0.5 * self._kl_threshold):
                lr = min(current_lr * 1.5, self._max_lr)
            self.model.learning_rate = lr
            self.model._setup_lr_schedule()
        # with torch.no_grad():
        #     self.model.policy.log_std.fill_(float(-10))


class SaveBestModel(BaseCallback):
    def __init__(self, log_dir, verbose=1):
        super().__init__()

        self.log_dir = log_dir
        self.save_path = os.path.join(self.log_dir, "best_model")
        self.best_rew = -np.inf
        self.verbose = verbose

        self.n_calls_rollout = 0

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self):
        return True
    
    def _on_rollout_end(self):
        # save best model
        self.n_calls_rollout += 1

        # Retrieve training reward
        if len(self.model.ep_info_buffer) > 0 and len(self.model.ep_info_buffer[0]) > 0:
            rew = safe_mean([ep_info["r"] for ep_info in self.model.ep_info_buffer])

            if rew > self.best_rew:
                self.best_rew = rew

                if self.verbose >= 1:
                    print(f"Saving new best model to {self.save_path}")

                model_path = os.path.join(self.save_path, "model")                
                self.model.save(model_path)
                if isinstance(self.model.env, VecNormalize):
                    env_path = os.path.join(self.save_path, "vec_normalize.pkl")  
                    self.model.env.save(env_path)


class DefaultRewardsShaper:
    def __init__(self, scale_value = 1, shift_value = 0, min_val=-np.inf, max_val=np.inf, is_torch=False):
        self.scale_value = scale_value
        self.shift_value = shift_value
        self.min_val = min_val
        self.max_val = max_val
        self.is_torch = is_torch

    def __call__(self, reward):
        
        reward = reward + self.shift_value
        reward = reward * self.scale_value
 
        if self.is_torch:
            reward = torch.clamp(reward, self.min_val, self.max_val)
        else:
            reward = np.clip(reward, self.min_val, self.max_val)
        return reward

@hydra.main(config_name="config", config_path="../cfg")
def parse_hydra_configs(cfg: DictConfig):

    time_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    headless = cfg.headless
    env = VecEnvSB3(headless=headless, sim_device=cfg.device_id)

    # ensure checkpoints can be specified as relative paths
    if cfg.checkpoint:
        cfg.checkpoint = retrieve_checkpoint_path(cfg.checkpoint)
        if cfg.checkpoint is None:
            quit()

    cfg_dict = omegaconf_to_dict(cfg)
    print_dict(cfg_dict)

    assert cfg_dict['algo'] == 'HER'

    task = initialize_task(cfg_dict, env)

    # sets seed. if seed is -1 will pick a random one
    from omni.isaac.core.utils.torch.maths import set_seed
    cfg.seed = set_seed(cfg.seed, torch_deterministic=cfg.torch_deterministic)

    if cfg.wandb_activate:
        # Make sure to install WandB if you actually use this.
        import wandb

        run_name = f"{cfg.wandb_name}_{time_str}"

        wandb.init(
            project=cfg.wandb_project,
            group=cfg.wandb_group,
            entity=cfg.wandb_entity,
            config=cfg_dict,
            sync_tensorboard=True,
            id=run_name,
            resume="allow",
            monitor_gym=True,
        )


    train_cfg = cfg_dict.get("train", dict())["params"]
    task_cfg = cfg_dict.get("task", dict())

    reward_shaper = DefaultRewardsShaper(**train_cfg["config"]["reward_shaper"])

    # wrap VecEnvBase (IsaacSim) to VecEnvWrapper (SB3)
    # env_wrapped = VecMonitorGPU(VecAdapter(env), device=cfg.rl_device)
    env_wrapped = VecMonitor(VecAdapter(env, reward_shaper=reward_shaper))
    # logger_path = "./logs/sb3_log_vfclip_wcoef_reward_shapping/"
    logger_path = "./logs/" + train_cfg["config"]["name"]

    # CONFIG
    model_class = DDPG
    online_sampling = True
    goal_selection_strategy = train_cfg["config"]["goal_selection_strategy"]
    max_episode_length = train_cfg["config"]["max_episode_length"]

    if not cfg.test: # train

        env_wrapped = VecNormalize(env_wrapped, training=True, clip_obs=10, norm_reward=False)

        activation_dict = {"elu": torch.nn.ELU, "relu": torch.nn.ReLU}
        activation_fn = activation_dict[train_cfg["network"]["mlp"]["activation"]]
        policy_kwargs = dict(activation_fn=activation_fn,
                            net_arch=train_cfg["network"]["mlp"]["units"],
                            log_std_init=0.0,
                            )

        replay_buffer_kwargs=dict(
                            n_sampled_goal=4, # K / HER sample ratio = 1 - 1/(k+1)
                            goal_selection_strategy=goal_selection_strategy,
                            online_sampling=online_sampling,
                            max_episode_length=max_episode_length,
                        ),
        # model = PPO(
        #         policy="MlpPolicy", 
        #         env=env_wrapped, 
        #         n_steps=train_cfg["config"]["horizon_length"], 
        #         learning_rate=train_cfg["config"]["learning_rate"], 
        #         batch_size=train_cfg["config"]["minibatch_size"], 
        #         n_epochs=train_cfg["config"]["mini_epochs"],
        #         gamma=train_cfg["config"]["gamma"],
        #         gae_lambda=train_cfg["config"]["tau"], #target_theta = (1-tau)*target_theta + tau*theta
        #         vf_coef=train_cfg["config"]["critic_coef"]/2,
        #         max_grad_norm=train_cfg["config"]["grad_norm"],
        #         clip_range_vf=train_cfg["config"]["e_clip"], # value error clip
        #         ent_coef=0.01, # entropy coefficient
        #         policy_kwargs=policy_kwargs,
        #         verbose=1,
        #         seed=cfg.seed,
        #         device=cfg.rl_device,
        #     )
        
        model = model_class(
            policy = "MultiInputPolicy", # support dict observation
            env = env_wrapped,
            replay_buffer_class=HerReplayBuffer,
            learning_rate=train_cfg["config"]["learning_rate"], 
            replay_buffer_kwargs=replay_buffer_kwargs, # param for HER
            buffer_size= 1e6, # default
            batch_size=train_cfg["config"]["minibatch_size"], #batch_size
            gamma=train_cfg["config"]["gamma"], #
            tau=train_cfg["config"]["tau"], #  #target_theta = (1-tau)*target_theta + tau*theta
            policy_kwargs=policy_kwargs,
            verbose=1,
            seed=cfg.seed,
            device=cfg.rl_device,
        )

        # set up logger
        new_logger = configure(logger_path)
        model.set_logger(new_logger)
        # horizon_length or episodeLength
        total_timesteps = env_wrapped.num_envs * train_cfg["config"]["horizon_length"] * train_cfg["config"]["max_epochs"]
        
        callback = [SaveBestModel(logger_path)]
        model.learn(total_timesteps=total_timesteps, progress_bar=False, callback=callback)

        model.save(logger_path + "/sb3_test_model")
        if isinstance(env_wrapped, VecNormalize):
            env_wrapped.save(logger_path + "/vec_normalize.pkl")

    else: # test
        from pathlib import Path
        ckpt_path = Path(cfg.checkpoint)
        env_wrapped = VecNormalize.load(ckpt_path.parent/"vec_normalize.pkl", env_wrapped)
        env_wrapped.training=False
        env_wrapped.norm_reward=False

        model = HER.load(ckpt_path)
        # env_wrapped._world.reset()
        obs = env_wrapped.reset()
        while env_wrapped._simulation_app.is_running():
            action, _states = model.predict(obs)
            obs, rewards, dones, info = env_wrapped.step(action)

    env_wrapped.close()



    if cfg.wandb_activate:
        wandb.finish()


if __name__ == '__main__':
    parse_hydra_configs()
