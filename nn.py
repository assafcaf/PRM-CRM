from math import ceil

import torch
from torch import nn
import numpy as np

class FullyConnectedMLP(nn.Module):
    """Vanilla two hidden layer multi-layer perceptron"""

    def __init__(self, obs_shape, h_size=64, emb_dim=32, n_actions=2, num_outputs=1):
        super(FullyConnectedMLP, self).__init__()
        input_dim = np.prod(obs_shape)
        self.double()
        self.embed = nn.Embedding(n_actions, emb_dim, max_norm=1)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim+emb_dim, h_size),
            nn.LeakyReLU(),
            nn.Linear(h_size, num_outputs),
            nn.Tanh()
        )

    def forward(self, obs, act):
        flat_obs = obs.flatten(1)
        emb = self.embed(act.long())
        x = torch.cat([flat_obs, emb], axis=1)
        return self.mlp(x)
    

class SimpleConvolveObservationQNet(FullyConnectedMLP):
    """
    Network that has two convolution steps on the observation space before flattening,
    concatinating the action and being an MLP.
    """

    def __init__(self, observation_space, features_dim=128, h_size=64, emb_dim=32, n_actions=2, num_outputs=1):
        super().__init__(features_dim, h_size, emb_dim, n_actions, num_outputs)
        # my backbonde
        self.back_bone = nn.Sequential(
            nn.Conv2d(observation_space.shape[0], 16, kernel_size=7, stride=3),
            nn.Dropout2d(0.5),
            # nn.BatchNorm2d(16),
            nn.LeakyReLU(0.01),

            nn.Conv2d(16, 16, kernel_size=5, stride=2),
            nn.Dropout2d(0.5),
            # nn.BatchNorm2d(16),
            nn.LeakyReLU(0.01),
            
            nn.Conv2d(16, 16, kernel_size=3, stride=1),
            nn.Dropout2d(0.5),
            # nn.BatchNorm2d(16),
            nn.LeakyReLU(0.01),
            
            nn.Conv2d(16, 16, kernel_size=3, stride=1),
            nn.Dropout2d(0.5),
            # nn.BatchNorm2d(16),
            nn.LeakyReLU(0.01),
            nn.Flatten()
        )
        
        # sb3 cnn
        # self.back_bone = nn.Sequential(
        # nn.Conv2d(observation_space.shape[0], 32, kernel_size=8, stride=4, padding=0),
        # nn.ReLU(),
        # nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
        # nn.ReLU(),
        # nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
        # nn.ReLU(),
        # nn.Flatten())
        
        with torch.no_grad():
            n_flatten = self.back_bone(torch.as_tensor(observation_space.sample()[None]).float()).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim),
                                    nn.ReLU())
        self.float()
        
    def forward(self, obs, act):
        # normalize the observation if it is not already
        if obs.max() > 1:
            obs = obs / 255.0
        
        # expand the observation if it is not already
        if len(obs.shape) == 3:
            # Need to add channels
            obs = torch.expand_dims(obs, axis=-1)
            
        emb = self.embed(act.view(-1).long())
        

        
        x = self.linear(self.back_bone(obs))
        x = torch.cat([x, emb], axis=1)
        return self.mlp(x)