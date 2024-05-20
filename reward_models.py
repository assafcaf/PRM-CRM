import random
from collections import deque


import numpy as np
import torch

from nn import FullyConnectedMLP, SimpleConvolveObservationQNet
from segment_sampling import sample_segment_from_path
from utils import corrcoef


def nn_predict_rewards(obs_segments, act_segments, network, obs_shape, act_shape):
    """
    :param obs_segments: tensor with shape = (batch_size, segment_length) + obs_shape
    :param act_segments: tensor with shape = (batch_size, segment_length) + act_shape
    :param network: neural net with .run() that maps obs and act tensors into a (scalar) value tensor
    :param obs_shape: a tuple representing the shape of the observation space
    :param act_shape: a tuple representing the shape of the action space
    :return: tensor with shape = (batch_size, segment_length)
    """
    # TODO: make this work with pytorch
    
    batchsize = (obs_segments).shape[0]
    segment_length = (obs_segments).shape[1]

    # Temporarily chop up segments into individual observations and actions
    # TODO: makesure its works fine without transpose (obs_shape)
    obs = obs_segments.view((-1,) + obs_shape)
    acts = act_segments.view((-1, 1))

    # # Run them through our neural network
    rewards = network(obs, acts)

    # # Group the rewards back into their segments
    # return tf.reshape(rewards, (batchsize, segment_length))
    return rewards.view((batchsize, segment_length))

class RewardModel(object):
    def __init__(self, episode_logger):
        self._episode_logger = episode_logger

    def predict_reward(self, path):
        raise NotImplementedError()  # Must be overridden

    def path_callback(self, path):
        self._episode_logger.log_episode(path)

    def train(self, iterations=1, report_frequency=None):
        pass  # Doesn't require training by default

    def save_model_checkpoint(self):
        pass  # Nothing to save

    def try_to_load_model_from_checkpoint(self):
        pass  # Nothing to load

class ComparisonRewardPredictor(RewardModel):
    """Predictor that trains a model to predict how much reward is contained in a trajectory segment"""

    def __init__(self, env, summary_writer, comparison_collector, agent_logger, label_schedule, stacked_frames, device, lr=0.0001, clip_length=0.1, train_freq=1e4):
        self.env = env
        self.summary_writer = summary_writer
        self.agent_logger = agent_logger
        self.comparison_collector = comparison_collector
        self.label_schedule = label_schedule
        self.stacked_frames = stacked_frames
        self.device = device
        
        # Set up some bookkeeping
        self.recent_segments = deque(maxlen=200)  # Keep a queue of recently seen segments to pull new comparisons from
        self._frames_per_segment = clip_length * env.fps
        self._steps_since_last_training = 0
        self._n_timesteps_per_predictor_training = train_freq  # How often should we train our predictor?
        self._elapsed_predictor_training_iters = 0

        # Build and initialize our predictor model
 
        self.obs_shape = (stacked_frames*env.observation_space.shape[0],) + env.observation_space.shape[1:]
        self.discrete_action_space = hasattr(env.action_space, "shape")
        self.act_shape = (env.action_space.n,) if self.discrete_action_space else env.action_space.shape
        
        self.model = self._build_model()
        self.loss = torch.nn.CrossEntropyLoss()
        self.train_op = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-4)
        
    def _predict_rewards(self, obs_segments, act_segments, network):
        """
        :param obs_segments: tensor with shape = (batch_size, segment_length) + obs_shape
        :param act_segments: tensor with shape = (batch_size, segment_length) + act_shape
        :param network: neural net with .run() that maps obs and act tensors into a (scalar) value tensor
        :return: tensor with shape = (batch_size, segment_length)
        """
        obs_segments = obs_segments.to(self.device).float()
        act_segments = act_segments.to(self.device)
        return nn_predict_rewards(obs_segments, act_segments, network, self.obs_shape, self.act_shape)

    def _build_model(self):
        """
        Our model takes in path segments with states and actions, and generates Q values.
        These Q values serve as predictions of the true reward.
        We can compare two segments and sum the Q values to get a prediction of a label
        of which segment is better. We then learn the weights for our model by comparing
        these labels with an authority (either a human or synthetic labeler).
        """
        # Set up action placeholder
        if self.discrete_action_space:
            # HACK Use a convolutional network for Atari
            # TODO Should check the input space dimensions, not the output space!
            net = SimpleConvolveObservationQNet(obs_shape=self.obs_shape, h_size=64, emb_dim=32, n_actions=self.act_shape[0])
             
        else:
            # In simple environments, default to a basic Multi-layer Perceptron (see TODO above)
            net = FullyConnectedMLP(obs_shape=self.obs_shape, h_size=64, emb_dim=32, n_actions=self.act_shape[0])


        # We use trajectory segments rather than individual (state, action) pairs because
        # video clips of segments are easier for humans to evaluate
        return net.to(self.device)

    def predict(self, obs, act):
        """Predict the reward for 1 time step """
        r = self.model(obs.float(), act.long().to(self.device))
        return r.cpu().detach().numpy().squeeze()
        
    def predict_reward(self, path):
        """Predict the reward for each step in a given path"""
        obs = path["obs"].to(self.device).float()
        action = path["actions"].to(self.device).long()

        return self.model(obs, action).detach().cpu()

    def path_callback(self, path):
        path_length = len(path["obs"])
        self._steps_since_last_training += path_length

        # TODO: write a proper summary writer for torch
        self.agent_logger.log_episode(path)

        # We may be in a new part of the environment, so we take new segments to build comparisons from
        segment = sample_segment_from_path(path, int(self._frames_per_segment))
        if segment and random.random() < 0.25:
            self.recent_segments.append(segment)

        # If we need more comparisons, then we build them from our recent segments
        if len(self.comparison_collector) < int(self.label_schedule.n_desired_labels):
            self.comparison_collector.add_segment_pair(
                random.choice(self.recent_segments),
                random.choice(self.recent_segments))

        # Train our predictor every X steps
        if self._steps_since_last_training >= int(self._n_timesteps_per_predictor_training):
            self.train_predictor()
            self._steps_since_last_training = 0

    def train_predictor(self, verbose=False):
        self.comparison_collector.label_unlabeled_comparisons()

        minibatch_size = min(8, len(self.comparison_collector.labeled_decisive_comparisons))
        labeled_comparisons = random.sample(self.comparison_collector.labeled_decisive_comparisons, minibatch_size)
        left_obs = np.asarray([comp['left']['obs'] for comp in labeled_comparisons])
        left_acts = np.asarray([comp['left']['actions'] for comp in labeled_comparisons])
        right_obs = np.asarray([comp['right']['obs'] for comp in labeled_comparisons])
        right_acts = np.asarray([comp['right']['actions'] for comp in labeled_comparisons])
        labels = np.asarray([comp['label'] for comp in labeled_comparisons])


        loss = self._train_step(left_obs, left_acts, right_obs, right_acts, labels).item()
        self._elapsed_predictor_training_iters += 1
        
        # TODO: write a proper summary writer for torch
        self._write_training_summaries(loss)
        
        if verbose:
            print("Reward predictor training iter %s (Err: %s)" % (self._elapsed_predictor_training_iters, loss))
            
        return loss
    
    def _train_step(self, left_obs, left_acts, right_obs, right_acts, labels):
        """ Train the model on a single batch """
        
        # move to torch
        left_obs = torch.tensor(np.stack(left_obs))
        left_acts = torch.tensor(left_acts)
                
        right_obs = torch.tensor(np.stack(right_obs))
        right_acts = torch.tensor(right_acts)
        
        labels = torch.tensor(labels).to(self.device)
        
        # predict rewards
        rewards_left = self._predict_rewards(left_obs, left_acts, self.model)
        rewards_right = self._predict_rewards(right_obs, right_acts, self.model)

        # We use trajectory segments rather than individual (state, action) pairs because
        # video clips of segments are easier for humans to evaluate
        segment_reward_pred_left = rewards_left.sum(axis=1, keepdim=True)
        segment_reward_pred_right = rewards_right.sum(axis=1, keepdim=True)
        reward_logits = torch.cat([segment_reward_pred_left, segment_reward_pred_right], axis=1) # (batch_size, 2)

        loss = self.loss(reward_logits, labels)
        
        self.train_op.zero_grad()
        loss.backward()
        self.train_op.step()
        torch.cuda.empty_cache()
        return loss
   
    def _write_training_summaries(self, loss):
        self.agent_logger.log_simple("predictor/_loss", loss)

        # Calculate correlation between true and predicted reward by running validation on recent episodes
        recent_paths = self.agent_logger.get_recent_paths_with_padding()
        if len(recent_paths) > 1 and self.agent_logger.summary_step % 10 == 0:  # Run validation every 10 iters
            idx = random.sample(range(len(recent_paths)), min(len(recent_paths), 10))
            validation_obs = np.asarray([path["obs"] for i, path in enumerate(recent_paths) if i in idx])
            validation_acts = np.asarray([path["actions"] for i, path in enumerate(recent_paths) if i in idx])
            q_value = self._predict_rewards(torch.from_numpy(validation_obs),
                                            torch.from_numpy(validation_acts),
                                            self.model).detach().cpu().numpy()
            ep_reward_pred = np.sum(q_value, axis=1)
            reward_true = np.asarray([path['original_rewards'] for i, path in enumerate(recent_paths) if i in idx])
            ep_reward_true = np.sum(reward_true, axis=1)
            self.agent_logger.log_simple("predictor/correlations", corrcoef(ep_reward_true, ep_reward_pred))
            torch.cuda.empty_cache()
            
        self.agent_logger.log_simple("predictor/num_training_iters", self._elapsed_predictor_training_iters)
        self.agent_logger.log_simple("labels/desired_labels", self.label_schedule.n_desired_labels)
        self.agent_logger.log_simple("labels/total_comparisons", len(self.comparison_collector))
        self.agent_logger.log_simple(
            "labels/labeled_comparisons", len(self.comparison_collector.labeled_decisive_comparisons))
    
    def buffer_usage(self):
        return self.comparison_collector.buffer_usage()
    

    class PRMComparisonRewardPredictor(RewardModel):
        def __init__(self, env, summary_writer, comparison_collector, agent_logger, label_schedule, stacked_frames, device, lr=0.0001, clip_length=0.1, train_freq=1e4):
            self.env = env
            self.summary_writer = summary_writer
            self.agent_logger = agent_logger
            self.comparison_collector = comparison_collector
            self.label_schedule = label_schedule
            self.stacked_frames = stacked_frames
            self.device = device
            
            # Set up some bookkeeping
            self.recent_segments = deque(maxlen=200)  # Keep a queue of recently seen segments to pull new comparisons from
            self._frames_per_segment = clip_length * env.fps
            self._steps_since_last_training = 0
            self._n_timesteps_per_predictor_training = train_freq  # How often should we train our predictor?
            self._elapsed_predictor_training_iters = 0

            # Build and initialize our predictor model

            self.obs_shape = (stacked_frames*env.observation_space.shape[0],) + env.observation_space.shape[1:]
            self.discrete_action_space = hasattr(env.action_space, "shape")
            self.act_shape = (env.action_space.n,) if self.discrete_action_space else env.action_space.shape
            
            self.model = self._build_model()
            self.loss = torch.nn.CrossEntropyLoss()
            self.train_op = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-4)