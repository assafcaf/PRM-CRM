from stable_baselines3 import PPO as sb3_PPO
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

from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.policies import ActorCriticCnnPolicy, ActorCriticPolicy, BasePolicy, MultiInputActorCriticPolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import explained_variance, get_schedule_fn
from stable_baselines3.common.utils import obs_as_tensor, safe_mean
import time
from collections import deque
from stable_baselines3.common import logger, utils 
from collections import OrderedDict


SelfPPO = TypeVar("SelfPPO", bound="PPO")


class PPO(sb3_PPO):
    def __init__(self,
                 policy: Union[str, Type[ActorCriticPolicy]],
                 env: Union[GymEnv, str],
                 learning_rate: Union[float, Schedule] = 3e-4,
                 n_steps: int = 2048,
                 batch_size: int = 64,
                 n_epochs: int = 10,
                 gamma: float = 0.99,
                 gae_lambda: float = 0.95,
                 clip_range: Union[float, Schedule] = 0.2,
                 clip_range_vf: Union[None, float, Schedule] = None,
                 normalize_advantage: bool = True,
                 ent_coef: float = 0.0,
                 vf_coef: float = 0.5,
                 max_grad_norm: float = 0.5,
                 use_sde: bool = False,
                 sde_sample_freq: int = -1,
                 target_kl: Optional[float] = None,
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
                         n_steps=n_steps,
                         batch_size=batch_size,
                         n_epochs=n_epochs,
                         gamma=gamma,
                         gae_lambda=gae_lambda,
                         clip_range=clip_range,
                         clip_range_vf=clip_range_vf,
                         normalize_advantage=normalize_advantage,
                         ent_coef=ent_coef,
                         vf_coef=vf_coef,
                         max_grad_norm=max_grad_norm,
                         use_sde=use_sde,
                         sde_sample_freq=sde_sample_freq,
                         target_kl=target_kl,
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
    
    def _setup_learn(
        self,
        total_timesteps: int,
        eval_env: Optional[GymEnv],
        callback: MaybeCallback = None,
        eval_freq: int = 10000,
        n_eval_episodes: int = 5,
        log_path: Optional[str] = None,
        reset_num_timesteps: bool = True,
        tb_log_name: str = "run",
    ):
        """
        Initialize different variables needed for training.

        :param total_timesteps: The total number of samples (env steps) to train on
        :param eval_env: Environment to use for evaluation.
        :param callback: Callback(s) called at every step with state of the algorithm.
        :param eval_freq: How many steps between evaluations
        :param n_eval_episodes: How many episodes to play per evaluation
        :param log_path: Path to a folder where the evaluations will be saved
        :param reset_num_timesteps: Whether to reset or not the ``num_timesteps`` attribute
        :param tb_log_name: the name of the run for tensorboard log
        :return:
        """
        self.start_time = time.time()

        if self.ep_info_buffer is None or reset_num_timesteps:
            # Initialize buffers if they don't exist, or reinitialize if resetting counters
            self.ep_info_buffer = deque(maxlen=100)
            self.ep_success_buffer = deque(maxlen=100)

        if self.action_noise is not None:
            self.action_noise.reset()

        if reset_num_timesteps:
            self.num_timesteps = 0
            self._episode_num = 0
        else:
            # Make sure training timesteps are ahead of the internal counter
            total_timesteps += self.num_timesteps
        self._total_timesteps = total_timesteps
        self._num_timesteps_at_start = self.num_timesteps




        # Configure logger's outputs if no logger was passed
        if not self._custom_logger:
            self._logger = utils.configure_logger(self.verbose, self.tensorboard_log, tb_log_name, reset_num_timesteps)

        # Create eval callback if needed
        callback = self._init_callback(callback, eval_env, eval_freq, n_eval_episodes, log_path)

        return total_timesteps, callback
    
    def add2buffer(self, observations, actions, rewards, dones, values, log_probs):
        self.rollout_buffer.add(observations,
                                    actions,
                                    rewards,
                                    dones,
                                    values,
                                    log_probs)

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

class IndependentPPO():
    def __init__(self, n_agents,
                 policy: Union[str, Type[ActorCriticPolicy]],
                 env: Union[GymEnv, str],
                 learning_rate: Union[float, Schedule] = 3e-4,
                 n_steps: int = 2048,
                 batch_size: int = 64,
                 n_epochs: int = 10,
                 gamma: float = 0.99,
                 gae_lambda: float = 0.95,
                 clip_range: Union[float, Schedule] = 0.2,
                 clip_range_vf: Union[None, float, Schedule] = None,
                 normalize_advantage: bool = True,
                 ent_coef: float = 0.0,
                 vf_coef: float = 0.5,
                 max_grad_norm: float = 0.5,
                 use_sde: bool = False,
                 sde_sample_freq: int = -1,
                 target_kl: Optional[float] = None,
                 tensorboard_log: Optional[str] = None,
                 create_eval_env: bool = False,
                 policy_kwargs: Optional[Dict[str, Any]] = None,
                 verbose: int = 0,
                 seed: Optional[int] = None,
                 device: Union[th.device, str] = "auto",
                _init_setup_model: bool = True):
        self.logger = None
        self.n_agents = n_agents
        self.num_timesteps = None
        
        # create empty vectiruze enc with apropriate params such as num_envs
        dummy_env = VecEnv(num_envs=env.num_envs//n_agents,
                           observation_space=env.observation_space,
                           action_space=env.action_space)
        
        self.agents = [PPO(policy=policy,
                           env=dummy_env,
                           learning_rate=learning_rate,
                           n_steps=n_steps,
                           batch_size=batch_size,
                           n_epochs=n_epochs,
                           gamma=gamma,
                           gae_lambda=gae_lambda,
                           clip_range=clip_range,
                           clip_range_vf=clip_range_vf,
                           normalize_advantage=normalize_advantage,
                           ent_coef=ent_coef,
                           vf_coef=vf_coef,
                           max_grad_norm=max_grad_norm,
                           use_sde=use_sde,
                           sde_sample_freq=sde_sample_freq,
                           target_kl=target_kl,
                           tensorboard_log=tensorboard_log,
                           create_eval_env=create_eval_env,
                           policy_kwargs=policy_kwargs,
                           verbose=verbose,
                           seed=seed,
                           device=device,
                           _init_setup_model=_init_setup_model)
                       for _ in range(n_agents)]
        
    def init_training(self, num_timesteps):
        for agent in self.agents:
            agent.init_training(num_timesteps)
    
    def after_train(self) -> None:
        agents_ep_info_buffer  = [agent.ep_info_buffer for agent in self.agents]
        
        self.iteration += 1
        self._update_current_progress_remaining(self.num_timesteps, self.total_timesteps)
        # Display training infos
        if self.agents[0].log_interval is not None and self.agents[0].iteration % self.agents[0].log_interval == 0:
            fps = int((self.agents[0].num_timesteps - self.agents[0]._num_timesteps_at_start) / (time.time() - self.start_time))
            self.logger.record("time/iterations", self.agents[0].iteration, exclude="tensorboard")
            if len(self.agents[0].ep_info_buffer) > 0 and len(self.agents[0].ep_info_buffer[0]) > 0:
                self.logger.record("rollout/ep_rew_mean", safe_mean([ep_info["r"] for ep_info_buffer in agents_ep_info_buffer for ep_info in ep_info_buffer]))
                self.logger.record("rollout/ep_len_mean", safe_mean([ep_info["l"] for ep_info_buffer in agents_ep_info_buffer for ep_info in ep_info_buffer]))
            self.logger.record("time/fps", fps)
            self.logger.record("time/time_elapsed", int(time.time() - self.agents[0].start_time), exclude="tensorboard")
            self.logger.record("time/total_timesteps", self.agents[0].num_timesteps, exclude="tensorboard")
            self.logger.dump(step=self.agents[0].num_timesteps)
            
            # social metrics
            self.logger.record("metrics/efficiency", safe_mean([ep_info["r"] for ep_info in self.agents[0].ep_info_buffer]))
            self.logger.record("metrics/equality", safe_mean([ep_info["equality"] for ep_info in self.agents[0].ep_info_buffer]))
            self.logger.record("metrics/sustainability", safe_mean([ep_info["sustainability"] for ep_info in self.agents[0].ep_info_buffer]))
            self.logger.record("metrics/peace", safe_mean([ep_info["peace"] for ep_info in self.agents[0].ep_info_buffer]))

    def _update_current_progress_remaining(self, num_timesteps: int, total_timesteps: int) -> None:
        for agent in self.agents:
            agent._update_current_progress_remaining(agent.num_timesteps, agent.total_timesteps)
    
    def get_policy(self):
        return self.policy
    
    def set_logger(self, logger):
        self.logger = logger
    
    def train(self):
        for agent in self.agents:
            agent.train()
    
    def add2buffer(self, agent_id, observations, actions, rewards, dones, values, log_probs):

        self.agents[agent_id].rollout_buffer.add(observations,
                                                 actions,
                                                 rewards,
                                                 dones,
                                                 values,
                                                 log_probs)

    def update_time_steps(self, time_steps):
        for agent in self.agents:
            agent.num_timesteps = time_steps
    
    @property
    def policies(self):
        return [agent.policy for agent in self.agents]
    
    def set_training_mode(self, mode=False):
        for agent in self.agents:
            agent.policy.set_training_mode(mode)

    def compute_returns_and_advantage(self, last_values, dones):
        for i, agent in enumerate(self.agents):
            self.rollout_buffer.compute_returns_and_advantage(last_values=last_values[i :: self.n_agents],
                                                              dones=dones[i :: self.n_agents])
    
    def _update_info_buffer(self, infos) -> None:
        for i, agent in enumerate(self.agents):
            agent._update_info_buffer(infos=infos[i :: self.n_agents])