import torch
from collections import deque
import random
import numpy as np
from scipy.spatial.distance import pdist
from scipy.special import softmax
from itertools import combinations
from torch.utils.data import Dataset, DataLoader


class Segment:
    def __init__(self, observations: torch.Tensor, actions: torch.Tensor, score: float):
        self.observations = observations
        self.actions = actions.to(torch.long)
        self.score = score

    def get_transition(self, device):
        return self.observations.to(device), self.actions.to(device)
 
class Record:
    def __init__(self, segment1: Segment, segment2: Segment, epsilon: float = 7):
        self.segment1 = segment1
        self.segment2 = segment2
        if np.abs((segment1.score - segment2.score)) > epsilon:
            # index 0 if segment1 better, 1 if segment2 better
            self.mu = int(segment1.score < segment2.score)
        else:
            self.mu = -1


class TransitionBuffer:
    def __init__(self, max_size=int(1e6), batch_size=64, epsilon=10.):
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)
        self.scores = deque(maxlen=max_size)
        self.batch_size = batch_size
        self.epsilon = epsilon

    def __len__(self):
        return len(self.buffer)

    def add(self, observations, actions, score):
        self.scores.append(score)
        self.buffer.append(Segment(observations, actions, score))

    def sample_uniform(self, batch_size=None):
        # records = []
        # for s1, s2 in zip(random.sample(self.buffer, self.batch_size), random.sample(self.buffer, self.batch_size)):
        #     record = Record(s1, s2, self.epsilon)
        #     records.append(record)
        # return records
        records = []
        while len(records) < self.batch_size:
            s1 = random.choice(self.buffer)
            s2 = random.choice(self.buffer)
            record = Record(s1, s2, self.epsilon)
            if record.mu != -1:
                records.append(record)
        return records
    
    def sample(self):
        d = pdist(np.expand_dims(np.array(self.scores), 1), metric="braycurtis")
        p = softmax(d, axis=0)
        pairs = list(combinations(range(len(self.scores)), 2))
        sample = random.choices(pairs, p, k=self.batch_size)
        records = []
        for s1, s2 in sample:
            record = Record(self.buffer[s1], self.buffer[s2], self.epsilon)
            # if .5 not in record.mu:
            records.append(record)
        return records

    def print_size(self):
        print(f"Buffer size {len(self) * self.buffer[0].__sizeof__() /2**30:.2f} GB with {len(self)} segments")