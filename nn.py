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

    def __init__(self, obs_shape, h_size=64, emb_dim=32, n_actions=2, num_outputs=1):
        after_convolve_shape = (
            int(ceil((ceil(obs_shape[1] - 6) / 3-4) / 2) -2 -2),
            int(ceil((ceil(obs_shape[2] - 6) / 3-4) / 2) -2 -2),
            16)
        super().__init__(after_convolve_shape, h_size, emb_dim, n_actions, num_outputs)
        #  original network
        # self.back_bone = nn.Sequential(
        #     nn.Conv2d(4, 16, kernel_size=7, stride=3),
        #     nn.Dropout2d(0.5),
        #     # nn.BatchNorm2d(16),
        #     nn.LeakyReLU(0.01),

        #     nn.Conv2d(16, 16, kernel_size=5, stride=2),
        #     nn.Dropout2d(0.5),
        #     # nn.BatchNorm2d(16),
        #     nn.LeakyReLU(0.01),
            
        #     nn.Conv2d(16, 16, kernel_size=3, stride=1),
        #     nn.Dropout2d(0.5),
        #     # nn.BatchNorm2d(16),
        #     nn.LeakyReLU(0.01),
            
        #     nn.Conv2d(16, 16, kernel_size=3, stride=1),
        #     nn.Dropout2d(0.5),
        #     # nn.BatchNorm2d(16),
        #     nn.LeakyReLU(0.01),
        # )
        
        # sb3 network
        self.back_bone = nn.Sequential(
            nn.Conv2d(obs_shape[0], 16, kernel_size=7, stride=3),
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
        )
        self.float()
        
    def forward(self, obs, act):
        if len(obs.shape) == 3:
            # Need to add channels
            obs = torch.expand_dims(obs, axis=-1)
        # Parameters taken from GA3C NetworkVP
        emb = self.embed(act.view(-1).long())
        
        if obs.max() > 1:
            obs = obs / 255.0
        
        # not sure why I need to transpose here but I do....
        # obs.transpose_(1, -1)
        x = self.back_bone(obs)
        x = x.flatten(1)
        x = torch.cat([x, emb], axis=1)
        return self.mlp(x)