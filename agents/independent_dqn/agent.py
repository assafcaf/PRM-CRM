from typing import Any, Dict, List, Optional, Tuple, Type, Union

import gym
import time
import numpy as np
import torch as th

from stable_baselines3 import DQN as sb3_DQN
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.dqn.policies import CnnPolicy, DQNPolicy, MlpPolicy, MultiInputPolicy
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.utils import obs_as_tensor, safe_mean, should_collect_more_steps
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, RolloutReturn, Schedule, TrainFreq, TrainFrequencyUnit
from stable_baselines3.common.noise import ActionNoise, VectorizedActionNoise

from agents.independent_dqn.buffer import PredictorBuffer
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.buffers import DictReplayBuffer, ReplayBuffer

import psutil 

#################################### ENV ORDER ####################################
# env order: [agent_1, ..., agent_n] * num_envs

class DummyGymEnv(gym.Env):
    def __init__(self, observation_space, action_space):
        self.observation_space = observation_space
        self.action_space = action_space

class IndependentDQN(sb3_DQN):
    def __init__(
        self,
        predictor,
        num_agents: int,
        real_rewards: bool,
        policy: Union[str, Type[DQNPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 1e-4,
        buffer_size: int = int(1e6),  
        learning_starts: int = 50000,
        batch_size: int = 32,
        tau: float = 1.0,
        gamma: float = 0.99,
        train_freq: Union[int, Tuple[int, str]] = 4,
        gradient_steps: int = 1,
        replay_buffer_class: Optional[ReplayBuffer] = ReplayBuffer,
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
                            train_freq=train_freq,
                            gradient_steps=gradient_steps,
                            replay_buffer_class=replay_buffer_class,
                            replay_buffer_kwargs=replay_buffer_kwargs,
                            optimize_memory_usage=optimize_memory_usage,
                            target_update_interval=target_update_interval,
                            create_eval_env=create_eval_env,
                            exploration_fraction=exploration_fraction,
                            seed=seed,
                            exploration_initial_eps=exploration_initial_eps,
                            gamma=gamma,
                            exploration_final_eps=exploration_final_eps,
                            max_grad_norm=max_grad_norm,
                            _init_setup_model=_init_setup_model,
                            policy_kwargs=policy_kwargs,
                            verbose=verbose,
                            device=device,
                            tensorboard_log=tensorboard_log)
        self.real_rewards = real_rewards
        self.num_agents = num_agents
        self.num_envs = env.num_envs // num_agents
        self.predictor = predictor

        
        
        env_fn = lambda: DummyGymEnv(self.observation_space, self.action_space)
        dummy_env = DummyVecEnv([env_fn] * self.num_envs)
        self.agents = [sb3_DQN(
                            policy=policy,
                            env=dummy_env,
                            learning_rate=learning_rate,
                            buffer_size=buffer_size,
                            learning_starts=learning_starts,
                            batch_size=batch_size,
                            tau=tau,
                            train_freq=train_freq,
                            gradient_steps=gradient_steps,
                            replay_buffer_class=replay_buffer_class,
                            replay_buffer_kwargs=replay_buffer_kwargs,
                            optimize_memory_usage=optimize_memory_usage,
                            target_update_interval=target_update_interval,
                            create_eval_env=create_eval_env,
                            exploration_fraction=exploration_fraction,
                            seed=seed,
                            exploration_initial_eps=exploration_initial_eps,
                            gamma=gamma,
                            exploration_final_eps=exploration_final_eps,
                            max_grad_norm=max_grad_norm,
                            _init_setup_model=_init_setup_model,
                            policy_kwargs=policy_kwargs,
                            verbose=0,
                            device=device)
                         for _ in range(self.num_agents)]

    def update_agents_last_obs(self):
        for i in range(self.num_agents):
            last_obs = self._last_obs[i::self.num_agents]
            self.agents[i]._last_obs = last_obs

    def learn(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 4,
        eval_env: Optional[GymEnv] = None,
        eval_freq: int = -1,
        n_eval_episodes: int = 5,
        tb_log_name: str = "run",
        eval_log_path: Optional[str] = None,
        reset_n_timesteps: bool = True,
    ) -> "OffPolicyAlgorithm":
    
        total_timesteps, callback = self._setup_learn(
            total_timesteps,
            eval_env,
            callback,
            eval_freq,
            n_eval_episodes,
            eval_log_path,
            reset_n_timesteps,
            tb_log_name,
        )
        self.update_agents_last_obs()
        for agent in self.agents:
            agent.set_logger(self.logger)

        callback.on_training_start(locals(), globals())

        while self.num_timesteps < total_timesteps:
            rollout = self.collect_rollouts(
                self.env,
                train_freq=self.agents[0].train_freq,
                action_noise=self.action_noise,
                callback=callback,
                learning_starts=self.learning_starts,
                replay_buffers=[agent.replay_buffer for agent in self.agents],
                log_interval=log_interval,
            )
            if rollout.continue_training is False:
                break

            if self.num_timesteps > 0 and self.num_timesteps > self.learning_starts:
                # If no `gradient_steps` is specified,
                # do as many gradients steps as steps performed during the rollout
                gradient_steps = self.gradient_steps if self.gradient_steps >= 0 else rollout.episode_timesteps
                # Special case when the user passes `gradient_steps=0`
                if gradient_steps > 0:
                    self.train(batch_size=self.batch_size, gradient_steps=gradient_steps)
            
            # check memory usage and if it exceeds 80% shut down the training 
            if psutil.virtual_memory().percent > 80:
                print("Memory usage is above 80%. Stopping training.")
                exit()
            
          
        callback.on_training_end()

        return self

    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        for agent in self.agents:
            agent.train(gradient_steps, batch_size)
    
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
     
    def _dump_logs(self) -> None:
        """
        Write log.
        """
        time_elapsed = time.time() - self.start_time
        fps = int((self.num_timesteps - self._num_timesteps_at_start) / (time_elapsed + 1e-8))
        self.logger.record("time/episodes", self._episode_num, exclude="tensorboard")
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
        self.logger.record("usage/dqn_buffer", self.agents[0].replay_buffer.size() * self.num_envs)
        self.logger.record("usage/dqn_buffer_percentage", self.agents[0].replay_buffer.size() / self.agents[0].buffer_size * self.num_envs)
        self.logger.record("usage/predictor_buffer", self.predictor.buffer_usage())
        
        self.logger.dump(step=self.num_timesteps)            
    
    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        train_freq: TrainFreq,
        replay_buffers: List[ReplayBuffer],
        action_noise: Optional[ActionNoise] = None,
        learning_starts: int = 0,
        log_interval: Optional[int] = None,
    ) -> RolloutReturn:
        """
        Collect experiences and store them into a ``ReplayBuffer``.

        :param env: The training environment
        :param callback: Callback that will be called at each step
            (and at the beginning and end of the rollout)
        :param train_freq: How much experience to collect
            by doing rollouts of current policy.
            Either ``TrainFreq(<n>, TrainFrequencyUnit.STEP)``
            or ``TrainFreq(<n>, TrainFrequencyUnit.EPISODE)``
            with ``<n>`` being an integer greater than 0.
        :param action_noise: Action noise that will be used for exploration
            Required for deterministic policy (e.g. TD3). This can also be used
            in addition to the stochastic policy for SAC.
        :param learning_starts: Number of steps before learning for the warm-up phase.
        :param replay_buffer:
        :param log_interval: Log data every ``log_interval`` episodes
        :return:
        """
        # Switch to eval mode (this affects batch norm / dropout)
        for agent in self.agents:
            agent.policy.set_training_mode(False)

        n_collected_steps, n_collected_episodes = 0, 0

        assert isinstance(env, VecEnv), "You must pass a VecEnv"
        assert train_freq.frequency > 0, "Should at least collect one step or episode."

        if env.num_envs > 1:
            assert train_freq.unit == TrainFrequencyUnit.STEP, "You must use only one env when doing episodic training."

        # Vectorize action noise if needed
        if action_noise is not None and env.num_envs > 1 and not isinstance(action_noise, VectorizedActionNoise):
            action_noise = VectorizedActionNoise(action_noise, env.num_envs)

        if self.use_sde:
            for agent in self.agents:
                agent.actor.reset_noise(env.num_envs)

        callback.on_rollout_start()
        continue_training = True

        while should_collect_more_steps(train_freq, n_collected_steps, n_collected_episodes):
            if self.use_sde and self.sde_sample_freq > 0 and n_collected_steps % self.sde_sample_freq == 0:
                # Sample a new noise matrix
                for agent in self.agents:
                    agent.actor.reset_noise(env.num_envs)

            # Select action randomly or according to policy
            agents_actions, agents_buffer_actions = [], []
            for agent in self.agents:
                actions_, buffer_actions_ = agent._sample_action(learning_starts, action_noise, self.num_envs)
                agents_actions.append(actions_)
                agents_buffer_actions.append(buffer_actions_)

            #  concatenate actions of all agents
            actions = np.concatenate(np.array(agents_actions).T, axis=0)
            buffer_actions = np.concatenate(np.array(agents_buffer_actions).T, axis=0)
            
            # Rescale and perform action
            new_obs, real_rewards, dones, infos = env.step(actions)
            
            # reward predictor
            if not self.real_rewards:
                pred_rewards = self.predictor.predict(obs_as_tensor(self._last_obs, self.policy.device),
                                                    th.tensor(actions).to(self.policy.device))
                try:
                    human_obs = self.env.get_images()
                except AttributeError:
                    human_obs = [info["human_obs"] for info in infos]
                self.predictor.store_step(self._last_obs, actions, pred_rewards, real_rewards, human_obs)
            else: pred_rewards = real_rewards
            
            
            self.num_timesteps += env.num_envs
            for agent in self.agents:
                 agent.num_timesteps += env.num_envs
            n_collected_steps += 1

            # Give access to local variables
            callback.update_locals(locals())
            # Only stop training if return value is False, not when it is None.
            if callback.on_step() is False:
                return RolloutReturn(n_collected_steps * env.num_envs, n_collected_episodes, continue_training=False)

            # Retrieve reward and episode length if using Monitor wrapper
            self._update_info_buffer(infos, dones)

            # Store data in replay buffer (normalized action and unnormalized observation)
            self.store_transitions(replay_buffers, buffer_actions, new_obs, pred_rewards, dones, infos)
            
            self._update_current_progress_remaining(self.num_timesteps, self._total_timesteps)
            for agent in self.agents:
                 agent._update_current_progress_remaining(agent.num_timesteps, self._total_timesteps)
            # For DQN, check if the target network should be updated
            # and update the exploration schedule
            # For SAC/TD3, the update is dones as the same time as the gradient update
            # see https://github.com/hill-a/stable-baselines/issues/900
            self._on_step()
            for agent in self.agents:
                 agent._on_step()
                 
            for idx, done in enumerate(dones):
                if done:
                    # Update stats
                    n_collected_episodes += 1
                    self._episode_num += 1
                    for agent in self.agents:
                        agent._episode_num += 1
                    

                    if action_noise is not None:
                        kwargs = dict(indices=[idx]) if env.num_envs > 1 else {}
                        action_noise.reset(**kwargs)


                    
            # train reward_predictor
            if dones.all():
            # Log training infos
                if log_interval is not None and self._episode_num % log_interval == 0:
                    self._dump_logs()
                if not self.real_rewards:
                    for (path, agent_id) in self.predictor.get_paths():
                        self.predictor.path_callback(path, agent_id) 

        callback.on_rollout_end()

        return RolloutReturn(n_collected_steps * env.num_envs, n_collected_episodes, continue_training)    

    def store_transitions(self, replay_buffers, actions, new_obs, rewards, dones, infos):
        self._last_obs = new_obs
        for i, (agent, replay_buffer) in enumerate(zip(self.agents, replay_buffers)):
            agent._store_transition(replay_buffer,
                                    actions[i::self.num_agents],
                                    new_obs[i::self.num_agents],
                                    rewards[i::self.num_agents],
                                    dones[i::self.num_agents],
                                    infos[i::self.num_agents])
    
    @classmethod
    def load(
        cls,
        path: str,
        policy: Union[str, Type[DQNPolicy]],
        num_agents: int,
        env: GymEnv,
        n_steps: int,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        tensorboard_log: Optional[str] = None,
        verbose: int = 0,
        **kwargs,
    ):
        model = cls(
            policy=policy,
            num_agents=num_agents,
            env=env,
            n_steps=n_steps,
            policy_kwargs=policy_kwargs,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            **kwargs,
        )
        env_fn = lambda: DummyGymEnv(env.observation_space, env.action_space)
        dummy_env = DummyVecEnv([env_fn] * (env.num_envs // num_agents))
        for polid in range(num_agents):
            model.policies[polid] = sb3_DQN.load(
                path=path + f"/policy_{polid + 1}/model", env=dummy_env, **kwargs
            )
        return model

    def save(self, path: str) -> None:
        for polid in range(self.num_agents):
            self.policies[polid].save(path=path + f"/policy_{polid + 1}/model")
    
    def set_up_agents(self):
        for agent in self.agents:
            agent.start_time = time.time()

            if agent.action_noise is not None:
                agent.action_noise.reset()

            agent.set_logger(self.logger)
            agent._last_episode_starts = np.ones((self.num_envs,), dtype=bool)

    def set_loggers(self, logger):
        for agent in self.agents:
            agent.set_logger(logger)
    
    def predict(self, observation, state=None, episode_start=None, deterministic=False):
        actions = np.zeros((self.num_envs*self.num_agents))
        for i, agent in enumerate(self.agents):
            ac, _ = agent.predict(observation[i::self.num_agents],
                                                        state,
                                                        episode_start,
                                                        deterministic)
            actions[i::self.num_agents] = ac
        return actions, None