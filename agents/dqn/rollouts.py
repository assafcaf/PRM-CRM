import torch
import numpy as np
from time import time
from time import sleep
import multiprocessing
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.utils import obs_as_tensor

class ParallelRollout(object):
    def __init__(self, vec_env, reward_predictor, learner, seed=42):
        self.predictor = reward_predictor
        self.env  = vec_env
        self.learner = learner
        self.policy =  self.learner.policy
        self.policy.set_training_mode(False)
        self._last_obs =  self.env.reset()
        self._episode_num = 0
         
    def rollout(self, n_frames, log_interval=int(1e3)):
        # sb3 params
        paths = [{'obs': [],
                  'actions': [],
                  'rewards': [],
                  'original_rewards': [],
                  'human_obs': []}
                 for _ in range(self.env.num_envs)] 
        
        self.learner.policy.set_training_mode(False)
        n_steps = 0
        self.learner.replay_buffer.reset()
        self.learner.callback.on_rollout_start()
        num_collected_episodes = 0
    
        # rolout params
        _last_episode_starts = np.ones(self.env.num_envs, dtype=bool)
        
        
        # rolout loop
        while n_steps < n_frames:
            actions, buffer_actions = self.learner._sample_action(learning_starts=self.learner.learning_starts, n_envs=self.env.num_envs)
            new_obs, real_rewards, dones, infos = self.env.step(actions.squeeze())
            
            # TODO: change to device maybe
            pred_rewards = self.predictor.predict(obs_as_tensor(self._last_obs, self.policy.device),
                                                  torch.tensor(actions).to(self.policy.device))
            
            self.learner.num_timesteps += self.env.num_envs

            # Give access to local variables
            self.learner.callback.update_locals(locals())
            if not self.learner.callback.on_step():
                return False

            self.learner._update_info_buffer(infos)
            n_steps += 1
            
            
            self.add2buffer(buffer_actions, new_obs, pred_rewards, dones, infos)
            self.learner._on_step()
            
            for idx, done in enumerate(dones):
                if done:
                    # Update stats
                    num_collected_episodes += 1
                    self._episode_num += 1

                    # Log training infos
                    if log_interval is not None and self._episode_num % log_interval == 0:
                        self.learner._dump_logs()
            try:
                human_obs = self.env.get_images()
            except AttributeError:
                human_obs = [info["human_obs"] for info in infos]
                
            for i in range(self.env.num_envs):
                paths[i]['obs'].append(self._last_obs[i])
                paths[i]['actions'].append(actions[i].item())
                paths[i]['rewards'].append(pred_rewards[i])
                paths[i]["original_rewards"].append(real_rewards[i])
                paths[i]["human_obs"].append(human_obs[i])
                
        self.learner.callback.on_rollout_end()
        return paths

    def add2buffer(self, buffer_actions, new_obs, rewards, dones, infos):
        self.learner.add2buffer(buffer_actions, new_obs, rewards, dones, infos)
        
class FilterOb:
    def __init__(self, filter_mean=True):
        self.m1 = 0
        self.v = 0
        self.n = 0.
        self.filter_mean = filter_mean

    def __call__(self, obs):
        self.m1 = self.m1 * (self.n / (self.n + 1)) + obs * 1 / (1 + self.n)
        self.v = self.v * (self.n / (self.n + 1)) + (obs - self.m1) ** 2 * 1 / (1 + self.n)
        self.std = (self.v + 1e-6)**.5  # std
        self.n += 1
        if self.filter_mean:
            o1 = (obs - self.m1) / self.std
        else:
            o1 = obs / self.std
        o1 = (o1 > 10) * 10 + (o1 < -10) * (-10) + (o1 < 10) * (o1 > -10) * o1
        return o1
    
filter_ob = FilterOb()