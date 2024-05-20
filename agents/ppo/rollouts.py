import torch
import numpy as np
from time import time
from time import sleep
import multiprocessing
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.utils import obs_as_tensor


def get_frame_stack(frames, depth):
    if depth < 1:
        # Don't stack
        return np.array(frames[-1])
    lookback_length = min(depth, len(frames) - 1)
    current_frame_copies = depth - lookback_length
    frames_list = [frames[-1] for _ in range(current_frame_copies)] + [frames[-i] for i in range(lookback_length)]
    # Reverse so the oldest frames come first instead of last
    frames_list.reverse()
    stacked_frames = np.array(frames_list)
    # Move the stack to be the last dimension and return
    
    # TODO: makesure its works fine without transpose
    # return np.transpose(stacked_frames, list(range(1, len(stacked_frames.shape))) + [0])
    return stacked_frames

    def __init__(self, env_id, make_env, stacked_frames, reward_predictor, num_workers, max_timesteps_per_episode, seed):
        self.num_workers = num_workers
        self.predictor = reward_predictor

        self.tasks_q = multiprocessing.JoinableQueue()
        self.results_q = multiprocessing.Queue()

        self.actors = []
        for i in range(self.num_workers):
            new_seed = seed * 1000 + i  # Give each actor a uniquely seeded env
            self.actors.append(Actor(
                self.tasks_q, self.results_q, env_id, make_env, stacked_frames, new_seed, max_timesteps_per_episode))

        for a in self.actors:
            a.start()

        # we will start by running 20,000 / 1000 = 20 episodes for the first iteration  TODO OLD
        self.average_timesteps_in_episode = 1000

    def rollout(self, timesteps):
        start_time = time()
        # keep 20,000 timesteps per update  TODO OLD
        # TODO Run by number of rollouts rather than time
        num_rollouts = int(timesteps / self.average_timesteps_in_episode)

        for _ in range(num_rollouts):
            self.tasks_q.put("do_rollout")
        self.tasks_q.join()

        paths = []
        for _ in range(num_rollouts):
            path = self.results_q.get()

            ################################
            #  START REWARD MODIFICATIONS  #
            ################################
            path["original_rewards"] = path["rewards"]
            path["rewards"] = self.predictor.predict_reward(path)
            self.predictor.path_callback(path)
            ################################
            #   END REWARD MODIFICATIONS   #
            ################################

            paths.append(path)

        self.average_timesteps_in_episode = sum([len(path["rewards"]) for path in paths]) / len(paths)

        return paths, time() - start_time

    def set_policy_weights(self, parameters):
        for i in range(self.num_workers):
            self.tasks_q.put(parameters)
        self.tasks_q.join()

    def end(self):
        for i in range(self.num_workers):
            self.tasks_q.put("kill")

class Actor_torch(multiprocessing.Process):
    def __init__(self, task_q, result_q, env_id, make_env, stacked_frames, seed, max_timesteps_per_episode, policy):
        multiprocessing.Process.__init__(self)
        self.env_id = env_id
        self.make_env = make_env
        self.stacked_frames = stacked_frames
        self.seed = seed
        self.task_q = task_q
        self.result_q = result_q
        self.max_timesteps_per_episode = max_timesteps_per_episode
        self.policy = policy
        self.policy.set_training_mode(False)

    # TODO Cleanup
    def set_policy(self, policy):
        self.policy.load_from_vector(policy)

    def act(self, obs):
        with torch.no_grad():
            obs = obs_as_tensor(np.expand_dims(obs, axis=0), self.policy.device)
            action, value, log_prob = self.policy(obs)
        return action.item(), value.item(), log_prob.item()

    def run(self):
        self.env = self.make_env(self.env_id, self.max_timesteps_per_episode)
        self.env.seed = self.seed

        self.continuous_actions = hasattr(self.env.action_space, "shape")

        # tensorflow variables (same as in model.py)
        observation_size = list(self.env.observation_space.shape)
        if self.stacked_frames > 0:
            observation_size += [self.stacked_frames]
        hidden_size = 64
        self.action_size = np.prod(self.env.action_space.shape) if self.continuous_actions else self.env.action_space.n

        # tensorflow model of the policy
        while True:
            next_task = self.task_q.get(block=True)
            if next_task == "do_rollout":
                # the task is an actor request to collect experience
                path = self.rollout()
                self.task_q.task_done()
                self.result_q.put(path)
            elif next_task == "kill":
                print("Received kill message for rollout process. Shutting down...")
                self.task_q.task_done()
                break
            else:
                # the task is to set parameters of the actor policy
                self.set_policy(next_task)

                # super hacky method to make sure when we fill the queue with set parameter tasks,
                # an actor doesn't finish updating before the other actors can accept their own tasks.
                sleep(0.1)
                self.task_q.task_done()

    def rollout(self):
        # TODO: change actor buffer to torch instead of append to lists
        unstacked_obs, obs, actions, rewards, dones = [], [], [], [], []
        values, log_probs, human_obs = [], [], []

        unstacked_obs.append(filter_ob(self.env.reset()))
        done = False
        last_episode_starts = True
        for i in range(self.max_timesteps_per_episode):
            ob = get_frame_stack(unstacked_obs, self.stacked_frames)
            action, value, log_prob = self.act(ob)

            obs.append(ob)
            actions.append(action)
            values.append(value)
            log_probs.append(log_prob)

            
            if self.continuous_actions:
                raw_ob, rew, done, info = self.env.step(action)
            else:
                choice = np.random.choice(self.action_size, p=action)
                raw_ob, rew, done, info = self.env.step(choice)



            unstacked_obs.append(filter_ob(raw_ob))
            rewards.append(rew)
            human_obs.append(info.get("human_obs"))
            dones.append(last_episode_starts)
            last_episode_starts = done
            
            if done:
                unstacked_obs.append(filter_ob(self.env.reset()))
                # terminal_obs = self.policy.obs_to_tensor(info["terminal_observation"])
                # terminal_value = self.policy.predict_values(terminal_obs) # type: ignore[arg-type]
                # rewards[-1] += terminal_value * 0.99
        
        # end of rolout
        ob = get_frame_stack(unstacked_obs, self.stacked_frames)
        _, next_value, _ = self.act(ob)
        path = {
            "obs":  torch.from_numpy(np.array(obs)).cpu(),
            "actions": torch.from_numpy(np.expand_dims(actions, -1)).cpu(),
            "values": torch.from_numpy(np.array(values)).cpu(),
            "log_probs":torch.from_numpy(np.array(log_probs)).cpu(),
            "rewards": torch.from_numpy(np.array(rewards)).cpu(),
            'dones': torch.from_numpy(np.array(dones)).cpu(),
            'last_values':  torch.from_numpy(np.array([next_value])),
            'last_dones': torch.from_numpy(np.array([done])),
            "human_obs": np.array(human_obs)}
        return path

class ParallelRollout_torch(object):
    def __init__(self, env_id, make_env, stacked_frames, reward_predictor, num_workers, max_timesteps_per_episode,
                 learner, seed=42):
        self.num_workers = num_workers
        self.predictor = reward_predictor

        self.tasks_q = multiprocessing.JoinableQueue()
        self.results_q = multiprocessing.Queue()

        self.actors = []
        for i in range(self.num_workers):
            new_seed = seed * 1000 + i  # Give each actor a uniquely seeded env
            self.actors.append(Actor_torch(
                self.tasks_q, self.results_q, env_id, make_env, stacked_frames, new_seed, max_timesteps_per_episode, learner.duplicat()))

        for a in self.actors:
            a.start()

        # we will start by running 20,000 / 1000 = 20 episodes for the first iteration  TODO OLD
        self.average_timesteps_in_episode = 1000

    def rollout(self, timesteps):
        start_time = time()
        # keep 20,000 timesteps per update  TODO OLD
        # TODO Run by number of rollouts rather than time
        num_rollouts = int(timesteps / self.average_timesteps_in_episode)

        for _ in range(num_rollouts):
            self.tasks_q.put("do_rollout")
        self.tasks_q.join()

        paths = []
        for _ in range(num_rollouts):
            path = self.results_q.get()

            ################################
            #  START REWARD MODIFICATIONS  #
            ################################
            path["original_rewards"] = path["rewards"]
            with torch.no_grad():
                path["rewards"] = self.predictor.predict_reward(path)
            self.predictor.path_callback(path)
            ################################
            #   END REWARD MODIFICATIONS   #
            ################################

            paths.append(path)

        self.average_timesteps_in_episode = sum([len(path["rewards"]) for path in paths]) / len(paths)

        return paths, time() - start_time

    def set_policy_weights(self, parameters):
        for i in range(self.num_workers):
            self.tasks_q.put(parameters)
        self.tasks_q.join()

    def end(self):
        for i in range(self.num_workers):
            self.tasks_q.put("kill")

class ParallelRollout(object):
    def __init__(self, vec_env, reward_predictor, learner, seed=42):
        self.predictor = reward_predictor
        self.env  = vec_env
        self.learner = learner
        self.policy =  self.learner.policy
        self.policy.set_training_mode(False)
        self._last_obs = self.env.reset()
        
    def act(self, obs):
        with torch.no_grad():
            obs = obs_as_tensor(obs, self.policy.device)
            action, value, log_prob = self.learner.policy(obs)
        return action.unsqueeze(-1).cpu(), value.cpu(), log_prob.cpu()
    
    def rollout(self, n_frames):
        # sb3 params
        paths = [{'obs': [],
                  'actions': [],
                  'rewards': [],
                  'original_rewards': [],
                  'human_obs': []}
                 for _ in range(self.env.num_envs)] 
        
        self.learner.policy.set_training_mode(False)
        n_steps = 0
        self.learner.rollout_buffer.reset()
        self.learner.callback.on_rollout_start()
        
        # rolout params
        _last_episode_starts = np.ones(self.env.num_envs, dtype=bool)
        done = False
        
        # rolout loop
        while n_steps < n_frames:
            actions, values, log_probs = self.act(self._last_obs)
            new_obs, real_rewards, dones, infos = self.env.step(actions.squeeze())
            
            # TODO: change to device maybe
            rewards = self.predictor.predict(obs_as_tensor(self._last_obs, self.policy.device), actions)
            self.learner.num_timesteps += self.env.num_envs

            # Give access to local variables
            self.learner.callback.update_locals(locals())
            if not self.learner.callback.on_step():
                return False

            self.learner._update_info_buffer(infos)
            n_steps += 1
            
            for idx, done in enumerate(dones):
                if (
                    done
                    and infos[idx].get("terminal_observation") is not None
                    and infos[idx].get("TimeLimit.truncated", False)
                ):
                    terminal_obs = self.policy.obs_to_tensor(infos[idx]["terminal_observation"])[0]
                    with torch.no_grad():
                        terminal_value = self.learner.policy.predict_values(terminal_obs)[0]  # type: ignore[arg-type]
                    rewards[idx] += self.learner.gamma * terminal_value
            self.learner.add2buffer(self._last_obs,  # type: ignore[arg-type]
                                    actions,
                                    rewards,
                                    _last_episode_starts,  # type: ignore[arg-type]
                                    values,
                                    log_probs,
                                )
            try:
                human_obs = self.env.get_images()
            except AttributeError:
                human_obs = [info["human_obs"] for info in infos]
                
            for i in range(self.env.num_envs):
                paths[i]['obs'].append(self._last_obs[i])
                paths[i]['actions'].append(actions[i].item())
                paths[i]['rewards'].append(rewards[i])
                paths[i]["original_rewards"].append(real_rewards[i])
                paths[i]["human_obs"].append(human_obs[i])
                
            self._last_obs = new_obs
            _last_episode_starts = dones
            
        # end of episode
        done = dones[0]
        with torch.no_grad():
            # Compute value for the last timestep
            values = self.learner.policy.predict_values(obs_as_tensor(new_obs, self.policy.device))  # type: ignore[arg-type]

        self.learner.compute_returns_and_advantage(last_values=values, dones=dones)
        self.learner.callback.update_locals(locals())
        self.learner.callback.on_rollout_end()
        return paths

class ParallelRolloutIndepentent(object):
    def __init__(self, vec_env, reward_predictor, learner, seed=42):
        self.predictor = reward_predictor
        self.env  = vec_env
        self.learner = learner
        self.learner.set_training_mode(False)
        self._last_obs = self.env.reset()
        self.device = self.learner.policies[0].device
        
    def act(self, obs):
        d = {}
        with torch.no_grad():
            for i, policy in enumerate(self.learner.policies):
                agent_obs = obs_as_tensor(obs[i :: self.learner.n_agents], self.device)
                action, value, log_prob = policy(agent_obs)
                d[i] = (action.squeeze().cpu(), value, log_prob)
        return d
    
    def rollout(self, n_frames):
        # sb3 params
        paths = [{'obs': [],
                  'actions': [],
                  'rewards': [],
                  'original_rewards': [],
                  'human_obs': []}
                 for _ in range(self.env.num_envs)] 
        
        for policy in self.learner.policies:
            policy.set_training_mode(False)
        n_steps = 0
        for agent in self.learner.agents:
            agent.rollout_buffer.reset()
            agent.callback.on_rollout_start()
        
        # rolout params
        _last_episode_starts = np.ones(self.env.num_envs, dtype=bool)
        done = False
        
        # rolout loop
        while n_steps < n_frames:
            # dict -> agent_id: (actions, values, log_probs)
            agents_outputs = self.act(self._last_obs)
            # stack actions for all agents first such taht [agent_1, agent_2, ..., agent_n, agent_1, agent_2, ...]
            actions = np.hstack(np.stack([action for action, _, _ in agents_outputs.values()], 1,))
            new_obs, real_rewards, dones, infos = self.env.step(actions.squeeze())
            
            # TODO: change to device maybe
            rewards = self.predictor.predict(obs_as_tensor(self._last_obs, self.device), torch.tensor(actions))
            
            self.learner.update_time_steps(self.env.num_envs)

            # Give access to local variables
            for agent in self.learner.agents:
                agent.callback.update_locals(locals())

            
            self.learner._update_info_buffer(infos)
            n_steps += 1
            
            for idx, done in enumerate(dones):
                if (
                    done
                    and infos[idx].get("terminal_observation") is not None
                    and infos[idx].get("TimeLimit.truncated", False)
                ):
                    terminal_obs = self.policy.obs_to_tensor(infos[idx]["terminal_observation"])[0]
                    with torch.no_grad():
                        ################################################### I'm here
                        terminal_value = np.zeros_like(rewards)
                        for i, policy in enumerate(self.learner.polysies):
                            terminal_value[i :: self.learner.n_agents] = policy.predict_values(terminal_obs[i :: self.learner.n_agents])[0] # type: ignore[arg-type]
                    rewards[idx] += self.learner.gamma * terminal_value
                    
            # add transitions to buffer per agent
            for i, agent in enumerate(self.learner.agents):
                actions, values, log_probs = agents_outputs[i]
                self.learner.add2buffer(i,
                                        self._last_obs[i :: self.learner.n_agents],  # type: ignore[arg-type]
                                        actions,
                                        rewards[i :: self.learner.n_agents],
                                        _last_episode_starts[i :: self.learner.n_agents],  # type: ignore[arg-type]
                                        values,
                                        log_probs,
                                    )
            try:
                human_obs = self.env.get_images()
            except AttributeError:
                human_obs = [info["human_obs"] for info in infos]
                
            for i in range(self.env.num_envs):
                paths[i]['obs'].append(self._last_obs[i])
                paths[i]['actions'].append(actions[i].item())
                paths[i]['rewards'].append(rewards[i])
                paths[i]["original_rewards"].append(real_rewards[i])
                paths[i]["human_obs"].append(human_obs[i])
                
            self._last_obs = new_obs
            _last_episode_starts = dones
            
        # end of episode
        done = dones[0]
        with torch.no_grad():
            # Compute value for the last timestep
            values = self.learner.policy.predict_values(obs_as_tensor(new_obs, self.device))  # type: ignore[arg-type]

        self.learner.compute_returns_and_advantage(last_values=values, dones=dones)
        self.learner.callback.update_locals(locals())
        self.learner.callback.on_rollout_end()
        return paths

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