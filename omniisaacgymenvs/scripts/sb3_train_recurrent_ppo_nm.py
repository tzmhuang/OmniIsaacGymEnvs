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
from omegaconf import DictConfig, OmegaConf

from stable_baselines3.common.logger import configure
# from stable_baselines3.common.vec_env import VecMonitorGPU
from stable_baselines3.common.vec_env import VecMonitor, VecNormalize, VecCheckNan, VecNormalizeCustom
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.utils import safe_mean
from stable_baselines3.common.save_util import load_from_pkl, save_to_pkl


from stable_baselines3.common.torch_layers import MlpFeatureExtractor , FlattenExtractor, MlpEncodeConcat
from sb3_contrib import RecurrentPPO


from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Type, Union

# from rl_games.common import env_configurations, vecenv
# from rl_games.torch_runner import Runner

import datetime
import os
import torch
import numpy as np
import json
 

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
    
    def __init__(self, active = True):
        super().__init__()
        self._learning_rate_start = None
        self._kl_threshold = 0.008
        self._min_lr = 1e-6
        self._max_lr = 1e-2

        self._active = active

    def _on_step(self):
        return True

    def _on_rollout_end(self) -> None:
        if not self._active:
            return
        # schedule lr
        kl_dist = self.model.approx_kl
        if kl_dist is not None:
            lr = current_lr = self.model.learning_rate
            if kl_dist > (2.0 * self._kl_threshold):
                print("up")
                lr = max(current_lr / 1.15, self._min_lr)
            if kl_dist < (0.5 * self._kl_threshold):
                lr = min(current_lr * 1.0, self._max_lr)
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
                if isinstance(self.model.env, VecNormalizeCustom):
                    env_path = os.path.join(self.save_path, "vec_normalize_custom.pkl")  
                    self.model.env.save(env_path)

                if hasattr(self.model, "replay_buffer") and self.model.replay_buffer is not None:
                    replay_buffer_path = os.path.join(model_path, "replay_buffer.pkl") 
                    save_to_pkl(replay_buffer_path, self.model.replay_buffer, False) 
                    if self.verbose > 1:
                        print(f"Saving model replay buffer checkpoint to {replay_buffer_path}")


class SaveIntermediateModel(BaseCallback):
    def __init__(self, log_dir, verbose=1, save_freq=200):
        super().__init__()

        self.log_dir = log_dir
        self.save_path = os.path.join(self.log_dir, "ckpt_model")
        # self.best_rew = -np.inf
        self.verbose = verbose
        self.save_freq = save_freq

        self.n_calls_rollout = 0

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self):
        return True
    
    def _on_rollout_end(self):
        self.n_calls_rollout += 1

        if len(self.model.ep_info_buffer) > 0 and len(self.model.ep_info_buffer[0]) > 0:

            if self.n_calls_rollout%self.save_freq == 0:# or self.n_calls_rollout in range(2190, 2200):   # NOTE: debug only

                if self.verbose >= 1:
                    print(f"Saving new best model to {self.save_path}")

                model_path = os.path.join(self.save_path, "iter_{}".format(self.n_calls_rollout))
                if model_path is not None:
                    os.makedirs(model_path, exist_ok=True)

                ckpt_path = os.path.join(model_path, "model")
                self.model.save(ckpt_path)

                if isinstance(self.model.env, VecNormalizeCustom):
                    env_path = os.path.join(model_path, "vec_normalize_custom.pkl")  
                    self.model.env.save(env_path)
                    if self.verbose >= 2:
                        print(f"Saving model VecNormalize to {vec_normalize_path}")

                if hasattr(self.model, "replay_buffer") and self.model.replay_buffer is not None:
                    replay_buffer_path = os.path.join(model_path, "replay_buffer.pkl") 
                    save_to_pkl(replay_buffer_path, self.model.replay_buffer, False) 
                    if self.verbose > 1:
                        print(f"Saving model replay buffer checkpoint to {replay_buffer_path}")


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

    task = initialize_task(cfg_dict, env)

    # sets seed. if seed is -1 will pick a random one
    from omni.isaac.core.utils.torch.maths import set_seed
    cfg.seed = set_seed(cfg.seed, torch_deterministic=cfg.torch_deterministic)
    print('-'*20)
    print('SEED Settings:')
    print('seed: ', cfg.seed)
    print('deterministic: ', cfg.torch_deterministic)
    print('-'*20)

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

    print(task_cfg)

    reward_shaper = DefaultRewardsShaper(**train_cfg["config"]["reward_shaper"])

    # wrap VecEnvBase (IsaacSim) to VecEnvWrapper (SB3)
    # env_wrapped = VecMonitorGPU(VecAdapter(env), device=cfg.rl_device)
    # env_wrapped = VecMonitor(VecAdapter(env, reward_shaper=reward_shaper))

    env_wrapped = VecCheckNan(VecMonitor(VecAdapter(env, reward_shaper=reward_shaper)), raise_exception=True) # NOTE: added
    # logger_path = "./logs/sb3_log_vfclip_wcoef_reward_shapping/"
    logger_path = "./logs/" + train_cfg["config"]["name"]

    # log config
    if not os.path.exists(logger_path):
        os.makedirs(logger_path)
    cfg_log_path = logger_path + "/config_log.yaml"
    OmegaConf.save(cfg, cfg_log_path)



    activation_dict = {"elu": torch.nn.ELU, "relu": torch.nn.ReLU}
    activation_fn = activation_dict[train_cfg["network"]["mlp"]["activation"]]

    if train_cfg["network"]["feature_extractor"]['type'] == None:
        features_extractor_class = FlattenExtractor
        features_extractor_dict = dict()
    elif train_cfg["network"]["feature_extractor"]['type'] == 'mlp':
        features_extractor_class = MlpFeatureExtractor
        feature_activation_fn = activation_dict[train_cfg["network"]["feature_extractor"]["feature_mlp"]["activation"]]
        features_extractor_dict = dict(
            net_arch=OmegaConf.to_container(train_cfg["network"]["feature_extractor"]["feature_mlp"]["units"]),
            activation_fn=feature_activation_fn,
            device=cfg.rl_device,
        )
    elif train_cfg["network"]["feature_extractor"]['type'] == 'encode_concat':
        features_extractor_class = MlpEncodeConcat
        feature_activation_fn = activation_dict[train_cfg["network"]["feature_extractor"]['encode_concat']["activation"]]
        features_extractor_dict = dict(
            loc_index=list(np.arange(2, 8)),
            proprio_index=list(np.arange(0,2))+list(np.arange(8, 8+21*2))+list(np.arange(8+21*3+24, 8+21*3+24+2)),    # no at_goal observation
            force_index=list(np.arange(8+21*2, 8+21*2+24)),
            action_index=list(np.arange(8+21*2+24, 8+21*3+24)),
            net_arch=OmegaConf.to_container(train_cfg["network"]["feature_extractor"]["encode_concat"]["units"]),
            activation_fn=feature_activation_fn,
            device=cfg.rl_device,
        )



    # if len(train_cfg["network"]["feature_mlp"]["units"]) == 0:
    #     features_extractor_class = FlattenExtractor
    #     features_extractor_dict = dict()
    # else:
    #     features_extractor_class = MlpFeatureExtractor
    #     feature_activation_fn = activation_dict[train_cfg["network"]["feature_mlp"]["activation"]]
    #     features_extractor_dict = dict(
    #         net_arch=OmegaConf.to_container(train_cfg["network"]["feature_mlp"]["units"]),
    #         activation_fn=feature_activation_fn,
    #         device=cfg.rl_device,
    #     )

    policy_kwargs = dict(
        # feature extractor settings
        features_extractor_class=features_extractor_class,
        features_extractor_kwargs=features_extractor_dict,
        # policy and value net settings
        activation_fn=activation_fn,
        net_arch=OmegaConf.to_container(train_cfg["network"]["mlp"]["units"]),
        log_std_init=0.0,
        # lstm settings
        lstm_hidden_size=train_cfg["network"]["lstm"]["hidden_size"],
        n_lstm_layers=train_cfg["network"]["lstm"]["n_layers"],
        shared_lstm = False, enable_critic_lstm = True,
        train_pred_at_goal = (not train_cfg["config"]["pretrain_walking"])
    )

    if train_cfg["config"]["enable_curiosity"]:
        curiosity_kwargs = dict(
            prediction_beta=train_cfg["config"]['curiosity_beta'],
            forward_loss_wt=train_cfg["config"]['curiosity_foward_weight'],
        )

        policy_kwargs.update(
            dict(
                enable_curiosity=train_cfg["config"]["enable_curiosity"],
                curiosity_kwargs = curiosity_kwargs
            )
        )

    if train_cfg["config"]["enable_ema"]:
        ema_kwargs = dict(
            n_envs=env_wrapped.num_envs,
            dim=train_cfg["network"]["lstm"]["hidden_size"],
            rew_weight=train_cfg["config"]['ema_weight'],
        )

        policy_kwargs.update(
            dict(
                enable_ema=train_cfg["config"]["enable_ema"],
                ema_kwargs = ema_kwargs
            )
        )




    if not cfg.test: # train

        from pathlib import Path
        
        if cfg.checkpoint:
            ckpt_path = Path(cfg.checkpoint)

            try:
                env_wrapped = VecNormalizeCustom.load(ckpt_path.parent/"vec_normalize_custom.pkl", env_wrapped)
            except FileNotFoundError:
                print('VecEnv Not Found, Creating from scratch')
                env_wrapped = VecNormalizeCustom(env_wrapped, training=True, norm_obs=True, skip_obs_idx=[2,3,5,6,7],clip_obs=10, norm_reward=train_cfg["config"]["norm_reward"])
        else:
            env_wrapped = VecNormalizeCustom(env_wrapped, training=True, norm_obs=True, skip_obs_idx=[2,3,5,6,7], clip_obs=10, norm_reward=train_cfg["config"]["norm_reward"])



        mini_batch_size = (train_cfg["config"]["horizon_length"] * env_wrapped.num_envs) // train_cfg["config"]["n_mini_batch"]

        model = RecurrentPPO(
                policy="MlpLstmPolicy", 
                env=env_wrapped, 
                n_steps=train_cfg["config"]["horizon_length"], 
                learning_rate=train_cfg["config"]["learning_rate"], 
                batch_size=mini_batch_size, 
                n_epochs=train_cfg["config"]["mini_epochs"],
                gamma=train_cfg["config"]["gamma"],
                gae_lambda=train_cfg["config"]["tau"],
                vf_coef=train_cfg["config"]["critic_coef"],
                max_grad_norm=train_cfg["config"]["grad_norm"],
                clip_range_vf=train_cfg["config"]["e_clip"], # value error clip
                ent_coef=train_cfg["config"]["entropy_coef"], # entropy coefficient
                policy_kwargs=policy_kwargs,
                verbose=1,
                seed=cfg.seed,
                device=cfg.rl_device,
            )

        if cfg.checkpoint:
            # param = model.get_parameters()

            model = RecurrentPPO.load(ckpt_path)
            # model.set_parameters(param)
            # print(model.seed)
            model.ent_coef = train_cfg["config"]["entropy_coef"]
            model.set_env(env_wrapped)
            model.set_random_seed(cfg.seed)

            model.policy.train_pred_at_goal = (not train_cfg["config"]["pretrain_walking"])


        # bound loss [deactivated]

        # model.policy.log_std.requires_grad_(False)

        # set up logger
        new_logger = configure(logger_path)
        model.set_logger(new_logger)
        
        # horizon_length or episodeLength
        total_timesteps = env_wrapped.num_envs * train_cfg["config"]["horizon_length"] * train_cfg["config"]["max_epochs"]

        if train_cfg["config"]["lr_schedule"] == "adaptive":
            use_adaptive = True
        elif train_cfg["config"]["lr_schedule"] == "fixed":
            use_adaptive = False
        else:
            raise ValueError("Unrecognized lr_schedule strategy {}".format(train_cfg["config"]["lr_schedule"]))
        
        callback = [CallbackHyperparamsSchedulePPO(use_adaptive), SaveBestModel(logger_path), SaveIntermediateModel(logger_path, save_freq=200)]
        model.learn(total_timesteps=total_timesteps, progress_bar=False, callback=callback)

        model.save(logger_path + "/final_model")
        if isinstance(env_wrapped, VecNormalizeCustom):
            env_wrapped.save(logger_path + "/vec_normalize_custom.pkl")

    else: # test
        from pathlib import Path
        ckpt_path = Path(cfg.checkpoint)

        try:
            env_wrapped = VecNormalizeCustom.load(ckpt_path.parent/"vec_normalize_custom.pkl", env_wrapped)
        except FileNotFoundError:
            env_wrapped = VecNormalizeCustom(env_wrapped, training=False, norm_obs=True, skip_obs_idx=[2,3,5,6,7], clip_obs=10, norm_reward=train_cfg["config"]["norm_reward"])


        env_wrapped.training=False
        env_wrapped.norm_reward=train_cfg["config"]["norm_reward"]

        model = RecurrentPPO.load(ckpt_path,)
        model.set_random_seed(cfg.seed)
        # env_wrapped._world.reset()
        obs = env_wrapped.reset()


        logger_path = "./testing_logs/" + train_cfg["config"]["name"]
        test_logger = configure(logger_path)
        model.set_logger(test_logger)

        timestep=0
        model.logger.record('test/max_obs', obs.max())
        model.logger.record('test/mean_obs', obs.mean())
        model.logger.record('test/min_obs', obs.min())
        model.logger.dump(step=timestep)



        # # lstm_states = None
        # num_envs = 1
        # # # Episode start signals are used to reset the lstm states
        # episode_starts = np.ones((env_wrapped.num_envs,), dtype=bool)
        # # while True:
        # #     action, lstm_states = model.predict(obs, state=lstm_states, episode_start=episode_starts, deterministic=True)
        # #     obs, rewards, dones, info = vec_env.step(action)
        # #     episode_starts = dones
        # #     vec_env.render("human")

        lstm_states = None
        episode_starts = np.ones((env_wrapped.num_envs,), dtype=bool)

        while env_wrapped._simulation_app.is_running():
            # action, lstm_states = model.predict(obs, state=lstm_states, episode_start=episode_starts, deterministic=True)
            agent_preds, lstm_states = model.predict(obs, state=lstm_states, episode_start=episode_starts, deterministic=True)
            action = agent_preds['actions']
            at_goal = agent_preds['at_goals']
            action = np.concatenate([action, at_goal], axis=-1)

            obs, rewards, dones, info = env_wrapped.step(action)        # [num_env, x]
            episode_starts = dones

            timestep += 1

            model.logger.record('test/max_obs', obs.max())
            model.logger.record('test/mean_obs', obs.mean())
            model.logger.record('test/min_obs', obs.min())
            # model.logger.record('test/action', action)
            model.logger.record('test/max_rewards', rewards.max())
            model.logger.record('test/mean_rewards', rewards.mean())
            model.logger.record('test/min_rewards', rewards.min())
            # model.logger.dump(step = timestep)
        


    env_wrapped.close()



    if cfg.wandb_activate:
        wandb.finish()


if __name__ == '__main__':
    parse_hydra_configs()
