import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import random
import numpy as np
from torch.nn.utils.rnn import pad_sequence
from scipy.spatial.distance import pdist
from scipy.special import softmax
from itertools import combinations
import time
from .network import ConvNetWork, ConvNetWork2, DenseNetwork, DenseNetworkTest
from .memory import TransitionBuffer


class RewardPredictorV2:
    def __init__(self, obs_dim, n_actions, hidden_dim=(256, 64, 16), lr=1e-4, batch_size=16, epsilon=5,
                 device="", epochs=4, max_buffer_size=10000, emb_dim=8):
        # init reward predictor network
        self.network = DenseNetworkTest(n_actions, channels=obs_dim[-1], emb_dim=emb_dim)
        self.network.to(device)
        self.loss = nn.NLLLoss(ignore_index=-1, reduction='mean')

        self.optimizer = optim.Adam(self.network.parameters(), lr=lr, weight_decay=5e-3)

        # transition buffer
        self.transition_buffer = TransitionBuffer(batch_size=batch_size, epsilon=epsilon, max_size=max_buffer_size)

        # hyperparameters
        self.device = device
        self.batch_size = batch_size
        self.epochs = epochs
        self.network.eval()
        
    def predict(self, observation, actions):
        return self.network(observation, actions).detach()
    
    def store_transition(self, observations, actions, score):
        self.transition_buffer.add(observations, actions, score)

    def learn(self):
        self.train(True)
        start = time.time()
        if len(self.transition_buffer) < self.batch_size:
            return 0, 0, 0
        ps, losses = [], []
        # self.network.train()
        transition_buffer = self.transition_buffer.sample_uniform()
        mu = torch.tensor([record.mu for record in transition_buffer]).to(self.device)
        minibatch_size = 16
        for _ in range(self.epochs):
            for idx in range(len(transition_buffer))[::minibatch_size]:
                minibatch_trainsition = transition_buffer[idx:idx+minibatch_size]
                minibatch_mu = mu[idx:idx+minibatch_size]
                p = torch.cat([self.ff(record.segment1.get_transition(self.device),
                                    record.segment2.get_transition(self.device))
                            for record in minibatch_trainsition]).view(minibatch_size, 2)
                loss = self.loss(p.log(), minibatch_mu)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                losses.append(loss.item())
                ps.append(p[torch.arange(minibatch_size), minibatch_mu][minibatch_mu > -1].mean().item())
                torch.cuda.empty_cache()
        return np.mean(losses), np.mean(ps), time.time() - start

    def ff(self, t1, t2):
        # r_pred = torch.cat([self.network(*t1), self.network(*t2)], dim=1)
        # # r_pred_normelized = r_pred / r_pred.std(axis=0) * 0.05
        # r_pred_normelized = r_pred

        # # mean_pred_r = r_pred_normelized.mean(axis=0)
        # mean_pred_r = r_pred_normelized.sum(axis=0) * (r_pred.shape[-1] ** -.5)
        # p = torch.clip(torch.softmax(mean_pred_r, dim=0), 0.001, 0.999)
        # return p
        
        mean_r_pred = torch.cat([self.network(*t1).mean(axis=0), self.network(*t2).mean(axis=0)], dim=1)
        p = torch.clip(torch.softmax(mean_r_pred, dim=0), 0.001, 0.999)
        return p

    def save(self, pth):
        torch.save(self.network.state_dict(), pth)

    def load(self, pth):
        self.network.load_state_dict(torch.load(pth))

    def train(self, flag):
        self.network.train(flag)

class RewardPredictor:
    '''
    Different reward predictor that learns with batches and epochs
    '''
    def __init__(self, obs_dim, n_actions, hidden_dim=(256, 64, 16), lr=1e-4, batch_size=16, epsilon=5,
                 device="", epochs=4, max_buffer_size=10000, emb_dim=8):
        # init reward predictor network
        self.network = ConvNetWork2(n_actions, channels=obs_dim[-1], emb_dim=emb_dim)
        self.network.to(device)
        self.loss = nn.NLLLoss(ignore_index=-1, size_average=True, reduction='sum')

        self.optimizer = optim.Adam(self.network.parameters(), lr=lr, weight_decay=5e-3)

        # transition buffer
        self.transition_buffer = TransitionBuffer(batch_size=batch_size, epsilon=epsilon, max_size=max_buffer_size)

        # hyperparameters
        self.device = device
        self.batch_size = batch_size
        self.epochs = epochs

    def predict(self, observation, actions):
        return self.network(observation, actions).detach()

    def store_transition(self, observations, actions, score):
        self.transition_buffer.add(observations, actions, score)

    def learn(self):
        self.train(True)
        start = time.time()
        if len(self.transition_buffer) < self.batch_size:
            return 0, 0, 0
        ps, losses = [], []
        self.network.train()
        transition_buffer = self.transition_buffer.sample_uniform()
        mu = torch.tensor([record.mu for record in transition_buffer]).to(self.device)

        for _ in range(self.epochs):
            
            p = torch.cat([self.ff(record.segment1.get_transition(self.device),
                                   record.segment2.get_transition(self.device))
                           for record in transition_buffer]).view(self.batch_size, 2)
            loss = self.loss(p.log(), mu)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            losses.append(loss.item())
            ps.append(p[torch.arange(self.batch_size), mu][mu > -1].mean().item())
        torch.cuda.empty_cache()
        self.network.eval()
        self.train(False)
        return np.mean(losses), np.mean(ps), time.time() - start

    def ff(self, t1, t2):
        r_pred = torch.cat([self.network(*t1), self.network(*t2)], dim=1)
        r_pred_normelized = r_pred / r_pred.std(axis=0) * 0.05
        # r_pred_normelized = r_pred

        # mean_pred_r = r_pred_normelized.mean(axis=0)
        mean_pred_r = r_pred_normelized.sum(axis=0) * (r_pred.shape[-1] ** -.5)
        p = torch.clip(torch.softmax(mean_pred_r, dim=0), 0.001, 0.999)
        return p

    def save(self, pth):
        torch.save(self.network.state_dict(), pth)

    def load(self, pth):
        self.network.load_state_dict(torch.load(pth))

    def train(self, flag):
        self.network.train(flag)


class PrmRP():
    def __init__(self, num_agents, n_actions, num_envs, obs_dim, hidden_dim=(256, 64, 16), lr=1e-4, batch_size=4, epsilon=5,
                 device="", epochs=4, max_buffer_size=1000, emb_dim=4, ep_length=1000):
        self.rps = [RewardPredictor(obs_dim, n_actions, hidden_dim, lr, batch_size, epsilon, device, epochs,
                                  max_buffer_size, emb_dim) for _ in range(num_agents)]
        self.ep_length = ep_length
        self.num_agents = num_agents
        self.num_envs = num_envs
        self.device = device
        self.emb_dim = emb_dim
        self.n_actions = n_actions
        self.trasitions_observation = torch.zeros((self.num_agents, self.num_envs, self.ep_length, *obs_dim[::-1]))
        self.trasitions_actions = torch.zeros((self.num_agents, self.num_envs, self.ep_length, 1))

    
    def predict(self, observation, actions):
        predictions = np.zeros((self.num_agents, self.num_envs))
        for agent in range(self.num_agents):
            # take only the observation and actions of the agent of interest assuming the folloing order:
            # [[agent_1, agents2 ... agent_n,] [agent_1, agents2 ... agent_n,] ...]
            indices = (torch.arange(len(observation)) % (self.num_agents)) == (agent)
            
            # predict the reward for the agent of interest and store it in the predictions array  assuming the folloing order:
            # [[agent_1_env1, agent_1_env2 ... env_n,] [agent_2_env1, agent_2_env2 ... env_n,] ...]
            predictions[agent] = self.rps[agent].predict(observation[indices], actions[indices]).cpu().numpy().T
        return predictions.flatten('F')
    
    def store(self, observation, actions, t):
        """ Store the observation and actions of the agent of interest in the transitions buffer. 
            The observation and actions are stored in the following order:
            [agents][envs][time_step]
        """
        for agent in range(self.num_agents):
            # take only the observation and actions of the agent of interest assuming the folloing order:
            # [[agent_1, agents2 ... agent_n,] [agent_1, agents2 ... agent_n,] ...]
            indices = (torch.arange(len(observation)) % (self.num_agents)) == (agent)
            self.trasitions_observation[agent][:, t] = observation[indices].cpu()
            self.trasitions_actions[agent][:, t] = actions[indices].cpu()
            
    def store_transition(self, scores):
        for agent in range(self.num_agents):
            agent_scors = scores[(torch.arange(len(scores)) % (self.num_agents)) == (agent)]
            for env in range(self.num_envs):
                # transition = [a for a in zip(self.trasitions_observation[agent][env],
                #                         self.trasitions_actions[agent][env])]
                self.rps[agent].store_transition(self.trasitions_observation[agent][env],
                                                 self.trasitions_actions[agent][env],
                                                 agent_scors[env])
    
    def learn(self):
         loss, ps, _ = zip(*[rp.learn() for rp in self.rps])
         return  loss, ps
    
    def train(self, flag):
        for rp in self.rps:
            rp.train(flag)
    
    def save(self, pth):
        for i, rp in enumerate(self.rps):
            rp.save(pth + f"_model_{i}")
    
    def load(self, pth):
        for i, rp in enumerate(self.rps):
            rp.load(pth + f"_model_{i}")

class SinglePrmRP():
    def __init__(self, n_actions, num_agents, obs_dim, hidden_dim=(512, 256, 16), lr=1e-3, batch_size=16, epsilon=5,
                 device="", epochs=10, max_buffer_size=10000, emb_dim=256, ep_length=1000):
        self.rp = RewardPredictor(obs_dim, n_actions, hidden_dim, lr, batch_size, epsilon, device, epochs,
                                   max_buffer_size, emb_dim)
        self.ep_length = ep_length
        self.num_agents = num_agents
        self.device = device
        self.emb_dim = emb_dim
        self.n_actions = n_actions
        self.trasitions_observation = torch.zeros((self.num_agents, self.ep_length, *obs_dim[::-1]))
        self.trasitions_actions = torch.zeros((self.num_agents, self.ep_length, 1))

    
    def predict(self, observation, actions):
        predictions = self.rp.predict(observation, actions).cpu().view(-1).numpy()
        return predictions
    
    def store(self, observation, actions, t):
        """ Store the observation and actions of the agent of interest in the transitions buffer. 
            The observation and actions are stored in the following order:
            [agents][envs][time_step]
        """
        for agent in range(self.num_agents):
            self.trasitions_observation[agent][t] = observation[agent].cpu()
            self.trasitions_actions[agent][t] = actions[agent].cpu()
            
    def store_transition(self, scores):
        for agent in range(self.num_agents):
            self.rp.store_transition(self.trasitions_observation[agent],
                                     self.trasitions_actions[agent],
                                     scores[agent])
    
    def learn(self):
         loss, ps, _ = self.rp.learn()
         return  loss, ps
    
    def train(self, flag):
        self.rp.train(flag)
    
    def save(self, pth):
        self.rp.save(pth + "_model")
    
    def load(self, pth):
        self.rp.load(pth + "_model")

