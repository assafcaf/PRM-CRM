from stable_baselines3 import DQN as sb3_DQN
from stable_baselines3.common.vec_env import VecEnv
from gym import spaces
import torch as th
import torch.nn.functional as F
import numpy as np

import warnings
from typing import Any, ClassVar, Dict, Optional, Type, TypeVar, Union, NamedTuple

import numpy as np
import torch as th
from gym import spaces
from torch.nn import functional as F

from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.dqn.policies import CnnPolicy, DQNPolicy, MlpPolicy, MultiInputPolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import explained_variance, get_schedule_fn
from stable_baselines3.common.utils import obs_as_tensor, safe_mean
import time
from collections import deque
from stable_baselines3.common import logger, utils 
from collections import OrderedDict
from typing import Tuple


SelfPPO = TypeVar("SelfPPO", bound="DQN")


class DQN(sb3_DQN):
    def __init__( self,
                 policy: Union[str, Type[DQNPolicy]],
                 env: Union[GymEnv, str],
                 learning_rate: Union[float, Schedule] = 1e-4,
                 buffer_size: int = 1_000_000,  # 1e6
                 learning_starts: int = 50000,
                 batch_size: int = 32,
                 tau: float = 1.0,
                 gamma: float = 0.99,
                 train_freq: Union[int, Tuple[int, str]] = 4,
                 gradient_steps: int = 1,
                 replay_buffer_class: Optional[ReplayBuffer] = None,
                 replay_buffer_kwargs: Optional[Dict[str, Any]] = None,
                 optimize_memory_usage: bool = False,
                 target_update_interval: int = 10000,
                 exploration_fraction: float = 0.1,
                 exploration_initial_eps: float = 1.0,
                 exploration_final_eps: float = 0.05,
                 max_grad_norm: float = 10,
                 tensorboard_log: Optional[str] = None,
                 create_eval_env: bool = False,
                 policy_kwargs: Optional[Dict[str, Any]] = None,
                 verbose: int = 0,
                 seed: Optional[int] = None,
                 device: Union[th.device, str] = "auto",
                 _init_setup_model: bool = True,
    ):
        super().__init__(policy=policy,
                         env=env,
                         learning_rate=learning_rate,
                         buffer_size=buffer_size,
                         learning_starts=learning_starts,
                         batch_size=batch_size,
                         tau=tau,
                         gamma=gamma,
                         train_freq=train_freq,
                         gradient_steps=gradient_steps,
                         replay_buffer_class=replay_buffer_class,
                         replay_buffer_kwargs=replay_buffer_kwargs,
                         optimize_memory_usage=optimize_memory_usage,
                         target_update_interval=target_update_interval,
                         exploration_fraction=exploration_fraction,
                         exploration_initial_eps=exploration_initial_eps,
                         exploration_final_eps=exploration_final_eps,
                         max_grad_norm=max_grad_norm,
                         tensorboard_log=tensorboard_log,
                         create_eval_env=create_eval_env,
                         policy_kwargs=policy_kwargs,
                         verbose=verbose,
                         seed=seed,
                         device=device,
                         _init_setup_model=_init_setup_model)
               
    def init_training(self,
                      total_timesteps: int,
                      callback: MaybeCallback = None,
                      log_interval: int = 1,
                      eval_env: Optional[GymEnv] = None,
                      eval_freq: int = -1,
                      n_eval_episodes: int = 5,
                      tb_log_name: str = "OnPolicyAlgorithm",
                      eval_log_path: Optional[str] = None,
                      reset_num_timesteps: bool = True) -> None:
        self.iteration = 0
        self.log_interval = log_interval
        self.total_timesteps, self.callback = self._setup_learn(
        total_timesteps, eval_env, callback, eval_freq, n_eval_episodes, eval_log_path, reset_num_timesteps, tb_log_name
        )
        self.callback.on_training_start(locals(), globals())
    
    def after_train(self) -> None:
        self.iteration += 1
        self._update_current_progress_remaining(self.num_timesteps, self.total_timesteps)
        # Display training infos
        if self.log_interval is not None and self.iteration % self.log_interval == 0:
            fps = int((self.num_timesteps - self._num_timesteps_at_start) / (time.time() - self.start_time))
            self.logger.record("time/iterations", self.iteration, exclude="tensorboard")
            if len(self.ep_info_buffer) > 0 and len(self.ep_info_buffer[0]) > 0:
                self.logger.record("rollout/ep_rew_mean", safe_mean([ep_info["r"] for ep_info in self.ep_info_buffer]))
                self.logger.record("rollout/ep_len_mean", safe_mean([ep_info["l"] for ep_info in self.ep_info_buffer]))
            self.logger.record("time/fps", fps)
            self.logger.record("time/time_elapsed", int(time.time() - self.start_time), exclude="tensorboard")
            self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
            self.logger.dump(step=self.num_timesteps)
            
            # social metrics
            self.logger.record("metrics/efficiency", safe_mean([ep_info["r"] for ep_info in self.ep_info_buffer]))
            self.logger.record("metrics/equality", safe_mean([ep_info["equality"] for ep_info in self.ep_info_buffer]))
            self.logger.record("metrics/sustainability", safe_mean([ep_info["sustainability"] for ep_info in self.ep_info_buffer]))
            self.logger.record("metrics/peace", safe_mean([ep_info["peace"] for ep_info in self.ep_info_buffer]))

    def get_policy(self):
        return self.policy
    
    def add2buffer(self, buffer_actions, new_obs, rewards, dones, infos):
      self._store_transition(self.replay_buffer,  buffer_actions, new_obs, rewards, dones, infos)

    def compute_returns_and_advantage(self, last_values, dones):
        self.rollout_buffer.compute_returns_and_advantage(last_values=last_values,
                                                            dones=dones)
    
    def _update_info_buffer(self, infos, dones=None) -> None:
        """
        Retrieve reward, episode length, episode success and update the buffer
        if using Monitor wrapper or a GoalEnv.

        :param infos: List of additional information about the transition.
        :param dones: Termination signals
        """
        if dones is None:
            dones = np.array([False] * len(infos))
        for idx, info in enumerate(infos):
            maybe_ep_info = info.get("episode")
            maybe_is_success = info.get("is_success")
            maybe_ep_metrics = info.get("metrics")
            if maybe_ep_info is not None:
                self.ep_info_buffer.extend([maybe_ep_info | maybe_ep_metrics])
            if maybe_is_success is not None and dones[idx]:
                self.ep_success_buffer.append(maybe_is_success)