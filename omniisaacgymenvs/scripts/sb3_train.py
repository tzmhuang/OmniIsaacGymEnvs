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

from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import VecMonitorGPU

from stable_baselines3.common.callbacks import BaseCallback, EvalCallback

from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Type, Union

# from rl_games.common import env_configurations, vecenv
# from rl_games.torch_runner import Runner

import datetime
import os
import torch

# class AdaptiveScheduler(RLScheduler):
#     def __init__(self, kl_threshold = 0.008):
#         super().__init__()
#         self.min_lr = 1e-6
#         self.max_lr = 1e-2
#         self.kl_threshold = kl_threshold

#     def update(self, current_lr, entropy_coef, epoch, frames, kl_dist, **kwargs):
#         lr = current_lr
#         if kl_dist > (2.0 * self.kl_threshold):
#             lr = max(current_lr / 1.5, self.min_lr)
#         if kl_dist < (0.5 * self.kl_threshold):
#             lr = min(current_lr * 1.5, self.max_lr)
#         return lr, entropy_coef    

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


# class CallbackEvalSchedulePPO(EvalCallback):
#     """Learning rate = Initial learning rate * training/std"""
    
#     def __init__(
#         self,
#         eval_env: Union[gym.Env, VecEnv],
#         callback_on_new_best: Optional[BaseCallback] = None,
#         callback_after_eval: Optional[BaseCallback] = None,
#         n_eval_episodes: int = 5,
#         eval_freq: int = 10000,
#         log_path: Optional[str] = None,
#         best_model_save_path: Optional[str] = None,
#         deterministic: bool = True,
#         render: bool = False,
#         verbose: int = 1,
#         warn: bool = True,
#     ):
#         super().__init__(
#             eval_env=eval_env,
#             callback_on_new_best=callback_on_new_best,
#             callback_after_eval=callback_after_eval,
#             n_eval_episodes=n_eval_episodes,
#             eval_freq=eval_freq,
#             log_path=log_path,
#             best_model_save_path=best_model_save_path,
#             deterministic=deterministic,
#             render=render,
#             verbose=verbose,
#             warn=warn, 
#         )
#         self._learning_rate_start = None

#     def _on_rollout_end(self) -> None:
#         if self._learning_rate_start is None:
#             self._learning_rate_start = self.model.learning_rate
#         std = torch.exp(self.model.policy.log_std).mean().item()
#         self.model.learning_rate = self._learning_rate_start * std
#         self.model._setup_lr_schedule()

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
        kl_dist = self.model.approx_kl
        if kl_dist is not None:
            lr = current_lr = self.model.learning_rate
            if kl_dist > (2.0 * self._kl_threshold):
                lr = max(current_lr / 1.5, self._min_lr)
            if kl_dist < (0.5 * self._kl_threshold):
                lr = min(current_lr * 1.5, self._max_lr)
            self.model.learning_rate = lr
            self.model._setup_lr_schedule()

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


    # wrap VecEnvBase (IsaacSim) to VecEnvWrapper (SB3)
    env_wrapped = VecMonitorGPU(VecAdapter(env), device=cfg.rl_device)
    logger_path = "./tmp/sb3_log/"

    if not cfg.test: # train
        train_cfg = cfg_dict.get("train", dict())["params"]
        task_cfg = cfg_dict.get("task", dict())

        activation_dict = {"elu": torch.nn.ELU, "relu": torch.nn.ReLU}
        activation_fn = activation_dict[train_cfg["network"]["mlp"]["activation"]]
        policy_kwargs = dict(activation_fn=activation_fn,
                            net_arch=train_cfg["network"]["mlp"]["units"],)


        model = PPO(
                policy="MlpPolicy", 
                env=env_wrapped, 
                n_steps=train_cfg["config"]["horizon_length"], 
                learning_rate=train_cfg["config"]["learning_rate"], 
                batch_size=train_cfg["config"]["minibatch_size"], 
                n_epochs=train_cfg["config"]["mini_epochs"],
                gamma=train_cfg["config"]["gamma"],
                gae_lambda=train_cfg["config"]["tau"],
                vf_coef=train_cfg["config"]["critic_coef"]/2,
                max_grad_norm=train_cfg["config"]["grad_norm"],
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
        learning_rate_callback = CallbackHyperparamsSchedulePPO()
        model.learn(total_timesteps=total_timesteps, progress_bar=True, callback=learning_rate_callback)

        model.save(logger_path + "sb3_test_model")

    else: # test
        model = PPO.load(cfg.checkpoint)
        # env_wrapped._world.reset()
        obs = env_wrapped.reset()
        while env._simulation_app.is_running():
            action, _states = model.predict(obs)
            obs, rewards, dones, info = env.step(action)
            obs = obs.cpu().numpy()

    env_wrapped.close()



    if cfg.wandb_activate:
        wandb.finish()


if __name__ == '__main__':
    parse_hydra_configs()
