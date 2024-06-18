from stable_baselines3 import PPO as sb3_PPO
import numpy as np
import gym
from typing import Any, Dict, List, Optional, Type, Union
import numpy as np
import torch as th
from stable_baselines3.common.policies import ActorCriticCnnPolicy, ActorCriticPolicy, BasePolicy, MultiInputActorCriticPolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import obs_as_tensor, safe_mean
import time
from collections import deque
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common import logger, utils 
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.buffers import DictRolloutBuffer, RolloutBuffer
import psutil 

class DummyGymEnv(gym.Env):
    def __init__(self, observation_space, action_space):
        self.observation_space = observation_space
        self.action_space = action_space
        
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

class IndependentPPO(sb3_PPO):
    def __init__(self,
                 predictor,
                 num_agents: int,
                 real_rewards: bool,
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
        
        self.real_rewards = real_rewards
        self.num_agents = num_agents
        self.num_envs = env.num_envs // num_agents
        self.predictor = predictor
        
        # create empty vectiruze enc with apropriate params such as num_envs
        env_fn = lambda: DummyGymEnv(self.observation_space, self.action_space)
        dummy_env = DummyVecEnv([env_fn] * self.num_envs)
        
        self.agents = [sb3_PPO(policy=policy,
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
                       for _ in range(self.num_agents)]
        
    def learn(
        self,
        total_timesteps: int,
        log_interval: int = 4,
        callback: Optional[List[MaybeCallback]] = None,
        progress_bar: bool = False,
        tb_log_name: str = "run",
        reset_num_timesteps: bool = True,
    ):

        # init agent params
        total_timesteps, callback = self._setup_learn(
                total_timesteps=total_timesteps,
                callback=callback,
                reset_num_timesteps=reset_num_timesteps,
                tb_log_name=tb_log_name,
                eval_env=None,
            )
        iteration = 0
        for agent in self.agents:
            agent.set_logger(self.logger)
        # init agents 
        self.update_agents_last_obs()
        self.update_last_episode_starts()
        for agent in self.agents:
                agent.set_logger(self.logger)
                
        callback.on_training_start(locals(), globals())

        assert self.env is not None
        
        # collect rollouts and training loop
        while self.num_timesteps < total_timesteps:
            continue_training = self.collect_rollouts(self.env,
                                                      callback=callback,
                                                      rollout_buffers=[agent.rollout_buffer for agent in self.agents],
                                                      n_rollout_steps=self.n_steps) 
            if not continue_training:
                break
            iteration += 1
            
            # update agents
            self._update_current_progress_remaining(self.num_timesteps, total_timesteps)
            for agent in self.agents:
                agent._update_current_progress_remaining(self.num_timesteps//self.num_agents,
                                                         total_timesteps//self.num_agents)
            
            # Display training infos
            if log_interval is not None and iteration % log_interval == 0:
                assert self.ep_info_buffer is not None
                self._dump_logs(iteration)
                
            self.train()

            callback.on_training_end()
            
            if psutil.virtual_memory().percent > 97:
                print(f"Memory usage at {psutil.virtual_memory().percent}%. Exiting...")
                exit()
        return self
    
    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        rollout_buffers: List[RolloutBuffer],
        n_rollout_steps: int,
    ) -> bool:
        """
        Collect experiences using the current policy and fill a ``RolloutBuffer``.
        The term rollout here refers to the model-free notion and should not
        be used with the concept of rollout used in model-based RL or planning.

        :param env: The training environment
        :param callback: Callback that will be called at each step
            (and at the beginning and end of the rollout)
        :param rollout_buffer: Buffer to fill with rollouts
        :param n_rollout_steps: Number of experiences to collect per environment
        :return: True if function returned with at least `n_rollout_steps`
            collected, False if callback terminated rollout prematurely.
        """
        assert self._last_obs is not None, "No previous observation was provided"
        # Switch to eval mode (this affects batch norm / dropout) and reset rollout_buffers
        for i, agent in enumerate(self.agents):
            agent.policy.set_training_mode(False)
            rollout_buffers[i].reset()

        n_steps = 0
        if self.use_sde:
            for i, agent in enumerate(self.agents):
                agent.policy.reset_noise(env.num_envs)
            self.policy.reset_noise(env.num_envs)
        callback.on_rollout_start()
        
        while n_steps < n_rollout_steps:
            if self.use_sde and self.sde_sample_freq > 0 and n_steps % self.sde_sample_freq == 0:
                # Sample a new noise matrix
                for i, agent in enumerate(self.agents):
                    agent.policy.reset_noise(env.num_envs)
                self.policy.reset_noise(env.num_envs)
                
            # predict actions
            all_clipped_actions, all_values, all_log_probs = self.feedforward(self._last_obs)
            actions = np.vstack(all_clipped_actions).transpose().reshape(-1) # reshape as (env, action)
            
            # env step
            new_obs, real_rewards, dones, infos = self.env.step(actions)
                
            # reward predictor
            if not self.real_rewards:
                rewards = self.predictor.predict(obs_as_tensor(self._last_obs, self.policy.device),
                                                    th.tensor(actions).to(self.policy.device))
                try:
                    human_obs = self.env.get_images()
                except AttributeError:
                    human_obs = [info["human_obs"] for info in infos]
                self.predictor.store_step(self._last_obs, actions, rewards, real_rewards, human_obs)
            else: rewards = real_rewards
            
            self.num_timesteps += env.num_envs
            for agent in self.agents:
                 agent.num_timesteps += env.num_envs
            
            # Give access to local variables
            callback.update_locals(locals())
            if not callback.on_step():
                return False
            
            # Retrieve reward and episode length if using Monitor wrapper
            self._update_info_buffer(infos, dones)
            n_steps += 1
            
             # Handle timeout by bootstraping with value function
            # see GitHub issue #633
            # TODO: make sure that the timeout is correctly handled
            # for idx, done in enumerate(dones):
            #     if (
            #         done
            #         and infos[idx].get("terminal_observation") is not None
            #         and infos[idx].get("TimeLimit.truncated", False)
            #     ):
            #         terminal_obs = self.policy.obs_to_tensor(infos[idx]["terminal_observation"])[0]
            #         with th.no_grad():
            #             terminal_value = self.policy.predict_values(terminal_obs)[0]  # type: ignore[arg-type]
            #         rewards[idx] += self.gamma * terminal_value
            for agent_id, rollout_buffer in enumerate(rollout_buffers):
                rollout_buffer.add(
                    self._last_obs[agent_id::self.num_agents],  # type: ignore[arg-type]
                    np.expand_dims(actions[agent_id::self.num_agents], -1),
                    rewards[agent_id::self.num_agents],
                    self._last_episode_starts[agent_id::self.num_agents],  # type: ignore[arg-type]
                    all_values[agent_id],
                    all_log_probs[agent_id]
                )
            self._last_obs = new_obs 
            self._last_episode_starts = dones
            
            self.update_agents_last_obs()
            self.update_last_episode_starts()
            
            
        _, values, _ = self.feedforward(self._last_obs)
        for i in range(self.num_agents):
            rollout_buffers[i].compute_returns_and_advantage(last_values=values[i], dones=dones[i::self.num_agents])

        # train reward_predictor
        if dones.all():
            if not self.real_rewards:
                for (path, agent_id) in self.predictor.get_paths():
                    self.predictor.path_callback(path, agent_id) 

        callback.update_locals(locals())
        callback.on_rollout_end()

        return True 
                 
    def feedforward(self, obs):
        all_actions = [None] * self.num_agents
        all_values = [None] * self.num_agents
        all_log_probs = [None] * self.num_agents
        all_clipped_actions = [None] * self.num_agents
        with th.no_grad():
                for agent_id, agent in enumerate(self.agents):
                    obs_tensor = obs_as_tensor(obs[agent_id::self.num_agents], agent.policy.device)
                    all_actions[agent_id], all_values[agent_id], all_log_probs[agent_id] = agent.policy(obs_tensor)
                    all_values[agent_id] = all_values[agent_id].view(self.num_envs)
                    clipped_actions = all_actions[agent_id].cpu().numpy()
                    all_clipped_actions[agent_id] = clipped_actions
        return all_clipped_actions, all_values, all_log_probs
    
    def train(self) -> None:
        for agent in self.agents:
            agent.train()
    
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
    
    # TODO adapt this method to PPO buffer
    def store_transitions(self, replay_buffers, actions, new_obs, rewards, dones, infos):
        self._last_obs = new_obs
        for i, (agent, replay_buffer) in enumerate(zip(self.agents, replay_buffers)):
            agent._store_transition(replay_buffer,
                                    actions[i::self.num_agents],
                                    new_obs[i::self.num_agents],
                                    rewards[i::self.num_agents],
                                    dones[i::self.num_agents],
                                    infos[i::self.num_agents])
    
    def set_loggers(self, logger):
        for agent in self.agents:
            agent.set_logger(logger)
    
    def _dump_logs(self, iteration: int) -> None:
        """
        Write log.
        """
        time_elapsed = time.time() - self.start_time
        fps = int((self.num_timesteps - self._num_timesteps_at_start) / (time_elapsed + 1e-8))
        self.logger.record("time/iteration", iteration, exclude="tensorboard")
        if len(self.ep_info_buffer) > 0 and len(self.ep_info_buffer[0]) > 0:
            self.logger.record("rollout/ep_rew_mean", safe_mean([ep_info["r"] for ep_info in self.ep_info_buffer]))
            self.logger.record("rollout/ep_len_mean", safe_mean([ep_info["l"] for ep_info in self.ep_info_buffer]))
            
            # social metrics
            self.logger.record("metrics/efficiency", safe_mean([ep_info["r"] for ep_info in self.ep_info_buffer]))
            self.logger.record("metrics/equality", safe_mean([ep_info["equality"] for ep_info in self.ep_info_buffer]))
            self.logger.record("metrics/sustainability", safe_mean([ep_info["sustainability"] for ep_info in self.ep_info_buffer]))
            self.logger.record("metrics/peace", safe_mean([ep_info["peace"] for ep_info in self.ep_info_buffer]))
        self.logger.record("time/fps", fps)
        self.logger.record("time/time_elapsed", int(time_elapsed), exclude="tensorboard")
        self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
        if self.use_sde:
            self.logger.record("train/std", (self.actor.get_std()).mean().item())

        if len(self.ep_success_buffer) > 0:
            self.logger.record("rollout/success_rate", safe_mean(self.ep_success_buffer))
        # Pass the number of timesteps for tensorboard
        
        self.logger.record("usage/memory", psutil.virtual_memory().percent)
        self.logger.record("usage/cpu", psutil.cpu_percent())
        self.logger.record("usage/predictor_buffer", self.predictor.buffer_usage())
        
        self.logger.dump(step=self.num_timesteps)
        if not self.real_rewards:
            self.predictor.dump(step=self.num_timesteps)         
    
    def update_agents_last_obs(self):
        for i in range(self.num_agents):
            last_obs = self._last_obs[i::self.num_agents]
            self.agents[i]._last_obs = last_obs
    
    def predict(self, observation, state=None, episode_start=None, deterministic=False):
        actions = np.zeros((self.num_envs*self.num_agents))
        for i, agent in enumerate(self.agents):
            ac, _ = agent.predict(observation[i::self.num_agents],
                                                        state,
                                                        episode_start,
                                                        deterministic)
            actions[i::self.num_agents] = ac
        return actions, None
    
    def update_last_episode_starts(self):
        for i in range(self.num_agents):
            last_episode_starts = self._last_episode_starts[i::self.num_agents]
            self.agents[i]._last_episode_starts = last_episode_starts