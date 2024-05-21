from social_dilemmas.envs.pettingzoo_env import parallel_env
from stable_baselines3.common.vec_env.vec_monitor import VecMonitor
from stable_baselines3.common.vec_env import VecTransposeImage
import supersuit as ss
from social_dilemmas.envs.wrappers import VecPrmEnv
from gym.spaces import Box
import numpy as np

def build_env(rollout_len, num_agents, num_cpus, num_frames, num_envs, use_my_wrap=True,
              metric=0, same_color=False, gray_scale=False, same_dim=False):
    env = parallel_env(
        max_cycles=rollout_len,
        env='harvest',
        num_agents=num_agents,
        ep_length=rollout_len,
        metric=metric,
        same_color=same_color,
        gray_scale=gray_scale,
        same_dim=same_dim
    )

    env = ss.observation_lambda_v0(env, lambda x, _: x["curr_obs"], lambda s: s["curr_obs"])
    env = ss.frame_stack_v1(env, num_frames)
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(
        env, num_vec_envs=num_envs, num_cpus=num_cpus, base_class="stable_baselines3"
    )
    env = VecTransposeImage(env)
    if use_my_wrap:
        env = _VecMonitor(env, rollout_len)
    else:
        env = VecMonitor(env)
    return env

class _VecMonitor(VecMonitor):
    def __init__(self, venv, rollout_len):
        super(_VecMonitor, self).__init__(venv)
        self.fps = rollout_len
        self._max_episode_steps = rollout_len
    
    def step(self, action):
        if type(action) != list:
            action = [action]
            ob, rew, done, info = super(_VecMonitor, self).step(action)
            return ob[0], rew[0], done[0], info[0]
        else:
          return super(_VecMonitor, self).step(action)

    def reset(self):
        ob = super(_VecMonitor, self).reset()
        return ob[0]