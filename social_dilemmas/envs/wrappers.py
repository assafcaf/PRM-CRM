from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvStepReturn, VecEnvWrapper
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
from .rp import PrmRP, SinglePrmRP

from collections import deque
import numpy as np
from tqdm import tqdm
import torch


class VecPrmEnv(VecEnvWrapper):
    """
    A vectorized wrapper for filtering a specific key from dictionary observations.
    Similar to Gym's FilterObservation wrapper:
        https://github.com/openai/gym/blob/master/gym/wrappers/filter_observation.py

    :param venv: The vectorized environment
    :param key: The key of the dictionary observation
    """

    def __init__(self, venv: VecEnv, num_envs, num_agents, device, num_frames, rp_kwargs, ep_length, 
                 singe=False):
        super().__init__(venv=venv)
        
        # init params
        self._num_envs = num_envs
        self.num_agents = num_agents
        self._num_frames = num_frames
        self.t = 0
        self.obs = self.reset()
        self.predicted_rewards = None
        self.done = None
        self.info = None
        self.iterations = 0
        self.pred_apples = []
        self.pred_fires = []
        self.pred_nothing = []
        self.rewards_deque = deque(maxlen=100)
        self.rewards_s = np.zeros(num_envs)
        
        # reward predictor
        if singe:
            self.rp = SinglePrmRP(obs_dim=self.observation_space.shape, 
                        num_agents=num_agents*num_envs,
                        n_actions=self.action_space.n,
                        ep_length=ep_length,
                        device=device, **rp_kwargs)
        else:
            self.rp = PrmRP(obs_dim=self.observation_space.shape, 
                            num_agents=num_agents,
                            num_envs=num_envs,
                            n_actions=self.action_space.n,
                            ep_length=ep_length,
                            device=device, **rp_kwargs)
        self.rewards_pred_s = np.zeros(num_envs)
        self.losses = deque(maxlen=100)
        self.ps = deque(maxlen=100)
        self.rs = deque(maxlen=100)
        self.ls = deque(maxlen=100)
        
    def reset(self) -> np.ndarray:
        self.obs = self.venv.reset()
        self.rewards_s = np.zeros(self.num_envs)
        self.t = 0
        self.pred_apples = []
        self.pred_fires = []
        self.pred_nothing = []
        self.rewards_pred_s = np.zeros(self.num_envs)
        self.transition = []
        return self.obs

    def step_async(self, actions: np.ndarray):
        obs, r, done, infos = self.step(actions)
        # rp predict
        observation_t, action_t = self.cat2tensor(obs, actions)
        with torch.no_grad():
            predicted_rewards = self.rp.predict(observation_t, action_t)

        # update infos
        for i in range(len(infos)):
            if infos[i]['r']:
                self.pred_apples.append(predicted_rewards[i])
            if infos[i]['fire']:
                self.pred_fires.append(predicted_rewards[i])
            if not infos[i]['fire'] and not infos[i]['r']:
                self.pred_nothing.append(predicted_rewards[i])
        
        # save transition
        self.obs = obs
        self.predicted_rewards = predicted_rewards
        self.done = done

        
        self.rewards_s += r
        self.t += 1
        # save transition reward predictor
        self.rp.store(observation_t, action_t, self.t-1)
        self.rewards_pred_s += predicted_rewards
        if done.any():
            self.rp.store_transition(self.rewards_s)
            loss, ps = self.rp.learn()
            self.losses.append(loss)
            self.ps.append(ps)
            self.rs.append(self.rewards_pred_s)
            self.ls.append(self.t)
            self.transition = []
            self.iterations += 1
            self.rewards_deque.append(self.rewards_s)
                    # update info
            for i in range(len(infos)):
                rp_data = {'rp_data': {'rp_loss': loss,
                                       'rp_ps': ps,
                                       'rp_rewards': predicted_rewards[i],
                                       'real_rewards': self.rewards_s[i],
                                       'pred_apples': np.nanmean(self.pred_apples),
                                       'pred_fires': np.nanmean(self.pred_fires),
                                       'pred_nothing': np.nanmean(self.pred_nothing),}
                         }
                # if sum([np.isnan(x) for x in rp_data['rp_data'].values()]):
                #     stop = 1
                #     print('nan')
                #     print(rp_data)
                infos[i] |= rp_data
            self.reset()
        self.info = infos
        return obs, predicted_rewards, done, self.info 
    
    def step(self, actions: np.ndarray) -> None:
        return self.venv.step(actions)

    def step_wait(self) -> VecEnvStepReturn:
        return self.obs, self.predicted_rewards, self.done, self.info
    
    def cat2tensor(self, obs, actions):
        """
        actions
        :param obs: observation from environment (np.ndarray) of shape (num_envs, obs_dim)
        :param action: action from environment (np.ndarray)
        :return: torch.tensor of shape (num_envs, obs_dim + 1)
        """
        obs = torch.tensor(obs, dtype=torch.float32, device=self.rp.device)
        actions = torch.tensor(actions, dtype=torch.long, device=self.rp.device).unsqueeze(dim=1)
        return obs.transpose(1, -1), actions

    def wormup(self, n_episodes):
        """
        :param num_steps: number of steps to wormup the reward predictor
        """
        self.rp.train(True)
        self.reset()
        for _ in tqdm(range(n_episodes), desc="Wormup", unit="episodes", ncols=80):
            done = False
            while not done:
                actions = [self.action_space.sample() for _ in range(self.num_agents*self._num_envs)]
                self.step_async(actions)
                done = self.done.all()
        self.rp.learn()
