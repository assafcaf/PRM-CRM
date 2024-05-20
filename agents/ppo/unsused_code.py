   
class RolloutData():
    def __init__(self,  gamma, gae_lambda, buffer_size, n_envs, device):
        
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.buffer_size = buffer_size
        self.n_envs = n_envs

        self.paths = None
        self.observations = None
        self.actions = None
        self.rewards = None
        self.values = None
        self.dones = None
        self.old_log_prob = None
        self.retuens = None
        self.advantages = None
        
        self.device = device
        self.generator_ready = False
        
        
    def feed(self, paths):
        self.generator_ready = False
        self.paths = paths
        self.observations = th.stack([p["obs"] for p in paths]).transpose(1, 0).float().cpu()
        self.actions = th.stack([p["actions"] for p in paths]).transpose(1, 0).long().cpu()
        self.rewards = th.stack([p["rewards"] for p in paths]).transpose(1, 0).float().cpu()
        self.values =  th.stack([p["values"] for p in paths]).transpose(1, 0).float().cpu()
        self.dones =  th.stack([p["dones"] for p in paths]).transpose(1, 0).float().cpu()
        self.log_probs =  th.stack([p["log_probs"] for p in paths]).transpose(1, 0).float().cpu()
        next_value = th.cat([p["next_values"] for p in paths]).float().cpu()
        # compute returns and advantage
        self.returns, self.advantages = self.compute_returns_and_advantage(next_value.numpy())
        self.generator_ready = False
    
    def compute_returns_and_advantage(self, next_value):
        rewards = self.rewards.numpy().squeeze()
        values = self.values.numpy().squeeze()
        dones = self.dones.numpy().squeeze()
        advantages = []
        last_gae_lam  = 0
        last_value = 0
        
        for step in reversed(range(len(rewards))):
            if step == len(rewards) - 1:
                next_non_terminal = 1.0 - dones[step]
                next_values = last_value
            else:
                next_non_terminal = 1.0 - dones[step + 1]
                next_values = values[step + 1]
                
            delta = rewards[step] + self.gamma * next_values * next_non_terminal - values[step]
            last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam 
            advantages.insert(0, last_gae_lam)
        advantages = np.array(advantages)
        returns = advantages + values
        return returns, advantages
    
    @staticmethod
    def swap_and_flatten(arr: np.ndarray) -> np.ndarray:
        """
        Swap and then flatten axes 0 (buffer_size) and 1 (n_envs)
        to convert shape from [n_steps, n_envs, ...] (when ... is the shape of the features)
        to [n_steps * n_envs, ...] (which maintain the order)

        :param arr:
        :return:
        """
        shape = arr.shape
        if len(shape) < 3:
            shape = shape + (1,)
        return arr.swapaxes(0, 1).reshape(shape[0] * shape[1], *shape[2:])
    
    def get(self, batch_size: Optional[int] = 64):
        indices = np.random.permutation(self.buffer_size * self.n_envs)
        # Prepare the data
        if not self.generator_ready:

            _tensor_names = [
                "observations",
                "actions",
                "values",
                "log_probs",
                "advantages",
                "returns",
            ]

            for tensor in _tensor_names:
                self.__dict__[tensor] = self.swap_and_flatten(self.__dict__[tensor])
            self.generator_ready = True

        # Return everything, don't create minibatches
        if batch_size is None:
            batch_size = self.buffer_size * self.n_envs

        start_idx = 0
        while start_idx < self.buffer_size * self.n_envs:
            yield self._get_samples(indices[start_idx : start_idx + batch_size])
            start_idx += batch_size
    
    def to_torch(self, array: np.ndarray, copy: bool = True) -> th.Tensor:
        """
        Convert a numpy array to a PyTorch tensor.
        Note: it copies the data by default

        :param array:
        :param copy: Whether to copy or not the data
        (may be useful to avoid changing things be reference)
        :return:
        """
        if copy:
            return th.tensor(array).to(self.device)
        return th.as_tensor(array).to(self.device)
    
    def _get_samples(self, batch_inds: np.ndarray):
        data = (
            self.observations[batch_inds],
            self.actions[batch_inds],
            self.values[batch_inds].flatten(),
            self.log_probs[batch_inds].flatten(),
            self.advantages[batch_inds].flatten(),
            self.returns[batch_inds].flatten(),
        )
        return RolloutBufferSamples(*tuple(map(self.to_torch, data)))

class RolloutBufferSamples(NamedTuple):
    observations: th.Tensor
    actions: th.Tensor
    old_values: th.Tensor
    old_log_prob: th.Tensor
    advantages: th.Tensor
    returns: th.Tensor
    
    

# model_torch
def duplicat(self):
    policy = self.policy_class(  # pytype:disable=not-instantiable
                self.observation_space,
                self.action_space,
                self.lr_schedule,
                use_sde=self.use_sde,
                **self.policy_kwargs  # pytype:disable=not-instantiable
            )
    policy = self.policy.to(self.device)
    return policy

def add2buffer(self, observations, actions, rewards, dones, values, log_probs):
    self.rollout_buffer.add(observations,
                                actions,
                                rewards,
                                dones,
                                values,
                                log_probs)

def compute_returns_and_advantage(self, last_values, dones):
    self.rollout_buffer.compute_returns_and_advantage(last_values=last_values,
                                                        dones=dones)

def insert2buffer(self, paths):
    self.rollout_buffer.reset()
    observations = th.stack([p["obs"] for p in paths]).transpose(1, 0).float().cpu()
    actions = th.stack([p["actions"] for p in paths]).transpose(1, 0).long().cpu()
    rewards = th.stack([p["rewards"] for p in paths]).transpose(1, 0).float().cpu()
    values =  th.stack([p["values"] for p in paths]).transpose(1, 0).float().cpu()
    dones =  th.stack([p["dones"] for p in paths]).transpose(1, 0).float().cpu()
    log_probs =  th.stack([p["log_probs"] for p in paths]).transpose(1, 0).float().cpu()
    
    
    last_values = th.cat([p["last_values"] for p in paths]).float().cpu()
    last_dones = th.cat([p["last_dones"] for p in paths]).float().cpu()

    for i in range(actions.shape[0]):
        self.rollout_buffer.add(observations[i],
                                actions[i],
                                rewards[i],
                                dones[i],
                                values[i],
                                log_probs[i])
        
    self.rollout_buffer.compute_returns_and_advantage(last_values=last_values,
                                                        dones=last_dones.numpy())




class OriginalEnvironmentReward(RewardModel):
    """Model that always gives the reward provided by the environment."""

    def predict_reward(self, path):
        return path["original_rewards"]

class OrdinalRewardModel(RewardModel):
    """A learned model of an environmental reward using training data that is merely sorted."""

    def __init__(self, model_type, env, env_id, make_env, experiment_name, episode_logger, label_schedule, n_pretrain_clips, clip_length, stacked_frames, workers):
        # TODO It's pretty asinine to pass in env, env_id, and make_env. Cleanup!
        super().__init__(episode_logger)

        if model_type == "synth":
            self.clip_manager = SynthClipManager(env, experiment_name)
        elif model_type == "human":
            self.clip_manager = ClipManager(env, env_id, experiment_name, workers)
        else:
            raise ValueError("Cannot find clip manager that matches keyword \"%s\"" % model_type)

        if self.clip_manager.total_number_of_clips > 0 and not self.clip_manager._sorted_clips:
            # If there are clips but no sort tree, create a sort tree!
            self.clip_manager.create_new_sort_tree_from_existing_clips()
        if self.clip_manager.total_number_of_clips < n_pretrain_clips:
            # If there aren't enough clips, generate more!
            self.generate_pretraining_data(env_id, make_env, n_pretrain_clips, clip_length, stacked_frames, workers)

        self.clip_manager.sort_clips(wait_until_database_fully_sorted=True)

        self.label_schedule = label_schedule
        self.experiment_name = experiment_name
        self._frames_per_segment = clip_length * env.fps
        # The reward distribution has standard dev such that each frame of a clip has expected reward 1
        self._standard_deviation = self._frames_per_segment
        self._elapsed_training_iters = 0
        self._episode_count = 0
        self._episodes_per_training = 50
        self._iterations_per_training = 50
        self._episodes_per_checkpoint = 100

        # Build and initialize our model
        self.obs_shape = env.observation_space.shape
        if stacked_frames > 0:
            self.obs_shape = self.obs_shape + (stacked_frames,)
        self.discrete_action_space = hasattr(env.action_space, "shape")
        self.act_shape = (env.action_space.n,) if self.discrete_action_space else env.action_space.shape

        # TODO: build a model for reward prediction
        self.model = self._build_model()

    def _build_model(self, lr=0.0001):
        """Our model takes in path segments with observations and actions, and generates rewards (Q-values)."""

        # Set up action placeholder
        if self.discrete_action_space:
            # HACK Use a convolutional network for Atari
            # TODO Should check the input space dimensions, not the output space!
            net = SimpleConvolveObservationQNet(self.obs_shape, self.act_shape, emb_dim=32, n_actions=self.act_shape[0])
        else:
            # In simple environments, default to a basic Multi-layer Perceptron (see TODO above)
            net = FullyConnectedMLP(self.obs_shape, self.act_shape, emb_dim=32, n_actions=self.act_shape[0])


        # We use trajectory segments rather than individual (state, action) pairs because
        # video clips of segments are easier for humans to evaluate

        self.loss = lambda segment_rewards, targets: torch.square(segment_rewards - targets).mean()
        self.train_op = torch.optim.Adam(net.parameters(), lr=lr)

        return net

    def predict_reward(self, pair):
        """Predict the reward for each step in a given path"""
        predicted_rewards = self.model(pair["obs"], pair["actions"])
        return predicted_rewards[0]  # The zero here is to get the single returned path.

    def path_callback(self, path):
        super().path_callback(path)
        self._episode_count += 1

        # We may be in a new part of the environment, so we take a clip to learn from if requested
        if self.clip_manager.total_number_of_clips < self.label_schedule.n_desired_labels:
            new_clip = sample_segment_from_path(path, int(self._frames_per_segment))
            if new_clip:
                self.clip_manager.add(new_clip, source="on-policy callback")

        # Train our model every X episodes
        if self._episode_count % self._episodes_per_training == 0:
            self.train(iterations=self._iterations_per_training, report_frequency=25)

        # Save our model every X steps
        if self._episode_count % self._episodes_per_checkpoint == 0:
            self.save_model_checkpoint()

    def generate_pretraining_data(self, env_id, make_env, n_pretrain_clips, clip_length, stacked_frames, workers):
        print("Starting random rollouts to generate pretraining segments. No learning will take place...")
        if self.clip_manager.total_number_of_clips == 0:
            # We need a valid clip for the root node of our search tree.
            # Null actions are more likely to generate a valid clip than a random clip from random actions.
            first_clip = basic_segment_from_null_action(env_id, make_env, clip_length, stacked_frames)
            # Add the null-action clip first, so the root is valid.
            self.clip_manager.add(first_clip, source="null-action", sync=True)  # Make synchronous to ensure this is the first clip.
            # Now add the rest

        desired_clips = n_pretrain_clips - self.clip_manager.total_number_of_clips

        random_clips = segments_from_rand_rollout(
            env_id, make_env, n_desired_segments=desired_clips,
            clip_length_in_seconds=clip_length, stacked_frames=stacked_frames, workers=workers)

        for clip in random_clips:
            self.clip_manager.add(clip, source="random rollout")

    def calculate_targets(self, ordinals):
        """ Project ordinal information into a cardinal value to use as a reward target """
        max_ordinal = self.clip_manager.maximum_ordinal  # Equivalent to the size of the sorting tree
        step_size = 1.0 / (max_ordinal + 1)
        offset = step_size / 2
        targets = [self._standard_deviation * stats.norm.ppf(offset + (step_size * o)) for o in ordinals]
        return targets

    def train(self, iterations=1, report_frequency=None):
        self.clip_manager.sort_clips()
        # batch_size = min(128, self.clip_manager.number_of_sorted_clips)
        _, clips, ordinals = self.clip_manager.get_sorted_clips()  # batch_size=batch_size

        obs = [clip['obs'] for clip in clips]
        acts = [clip['actions'] for clip in clips]
        targets = self.calculate_targets(ordinals)


        for i in range(1, iterations + 1):
            loss = self._train_step(obs, acts, targets)
            self._elapsed_training_iters += 1
            if report_frequency and i % report_frequency == 0:
                print("%s/%s reward model training iters. (Err: %s)" % (i, iterations, loss))
            elif iterations == 1:
                print("Reward model training iter %s (Err: %s)" % (self._elapsed_training_iters, loss))
        pass
    
    def _train_step(self, obs, act, targets):
        """ Train the model on a single batch """
        obs = torch.tensor(np.stack(obs))
        act = torch.tensor(act)
        targets = torch.tensor(targets)
        
        rewards = nn_predict_rewards(obs, act, self.model, self.obs_shape, self.act_shape)
        segment_rewards = rewards.sum(axis=1)
        loss = self.loss(targets, segment_rewards)
        
        self.train_op.zero_grad()
        loss.backward()
        self.train_op.step()
        return loss.item()
        
    def _checkpoint_filename(self):
        return 'checkpoints/reward_model/%s/treesave' % (self.experiment_name)

    def save_model_checkpoint(self):
        # TODO: write a proper checkpoint saver for torch
        # print("Saving reward model checkpoint!")
        # self.saver.save(self.sess, self._checkpoint_filename())
        pass

    def try_to_load_model_from_checkpoint(self):
        # TODO: write a proper checkpoint loader for torch
        # filename = tf.train.latest_checkpoint(os.path.dirname(self._checkpoint_filename()))
        # if filename is None:
        #     print('No reward model checkpoint found on disk for experiment "{}"'.format(self.experiment_name))
        # else:
        #     self.saver.restore(self.sess, filename)
        #     print("Reward model loaded from checkpoint!")
        #     # Dump model outputs with errors
        #     if True:  # <-- Toggle testing with this
        #         with self.graph.as_default():
        #             clip_ids, clips, ordinals = self.clip_manager.get_sorted_clips()
        #             targets = self.calculate_targets(ordinals)
        #             for i in range(len(clips)):
        #                 predicted_rewards = self.sess.run(self.rewards, feed_dict={
        #                     self.obs_placeholder: np.asarray([clips[i]["obs"]]),
        #                     self.act_placeholder: np.asarray([clips[i]["actions"]]),
        #                     K.learning_phase(): False
        #                 })[0]
        #                 reward_sum = sum(predicted_rewards)
        #                 starting_reward = predicted_rewards[0]
        #                 ending_reward = predicted_rewards[-1]
        #                 print(
        #                     "Clip {: 3d}: predicted = {: 5.2f} | target = {: 5.2f} | error = {: 5.2f}"  # | start = {: 5.2f} | end = {: 5.2f}"
        #                     .format(clip_ids[i], reward_sum, targets[i], reward_sum - targets[i]))  # , starting_reward, ending_reward))
        pass
    