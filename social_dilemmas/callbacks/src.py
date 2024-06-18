from stable_baselines3.common.callbacks import BaseCallback
import numpy as np
import cv2
import imageio
import os
import json

class IndependentAgentCallback(BaseCallback):
    """
    A custom callback that derives from ``BaseCallback``.

    :param verbose: (int) Verbosity level 0: not output 1: info 2: debug
    """
    def __init__(self, eval_env, verbose=0, render_frequency=10, deterministic=False, video_resolution=(720, 480)):
        super(IndependentAgentCallback, self).__init__(verbose)
        # Those variables will be accessible in the callback
        # (they are defined in the base class)
        # The RL model
        # self.model = None  # type: BaseRLModel
        # An alias for self.model.get_env(), the environment used for training
        # self.training_env = None  # type: Union[gym.Env, VecEnv, None]
        # Number of time the callback was called
        # self.n_calls = 0  # type: int
        # self.num_timesteps = 0  # type: int
        # local and global variables
        # self.locals = None  # type: Dict[str, Any]
        # self.globals = None  # type: Dict[str, Any]
        # The logger object, used to report things in the terminal
        # self.logger = None  # type: logger.Logger
        # # Sometimes, for event callback, it is useful
        # # to have access to the parent object
        # self.parent = None  # type: Optional[BaseCallback]
        self.iterations_ = 0
        self.render_frequency = render_frequency
        self.eval_env = eval_env
        self.deterministic = deterministic
        self.video_resolution = video_resolution
    def _on_training_start(self) -> None:
        file_name = os.path.join(self.model.logger.dir, "parameters.json")

        params = {"learning_rate": self.model.agents[0].learning_rate,
                  "batch_size": self.model.agents[0].batch_size,
                  "gae_lambda": self.model.agents[0].gae_lambda,
                  "gamma": self.model.agents[0].gamma,
                  "n_envs": self.model.agents[0].n_envs,
                  "n_epochs": self.model.agents[0].n_epochs,
                  "normalize_advantage": self.model.agents[0].normalize_advantage,
                  "target_kl": self.model.agents[0].target_kl,
                  "ent_coef": self.model.agents[0].ent_coef,
                  # each frame consist from 3 channels (RGB)
                  "n_frames": self.model.env.observation_space.shape[0],
                  "policy_kwargs": self.model.agents[0].policy.features_extractor_kwargs,
                  "policy_type": str(type(self.model.agents[0].policy.features_extractor)).split(".")[-1],
                  "observations_space": str(self.model.agents[0].observation_space)}

        json_object = json.dumps(params, indent=4)
        with open(file_name, "w") as outfile:
            outfile.write(json_object)

    def _on_rollout_start(self) -> None:
        """
        A rollout is the collection of environment interaction
        using the current policy.
        This event is triggered before collecting new samples.
        """
        pass

    def _on_step(self) -> bool:
        """
            This method will be called by the model after each call to `env.step()`.

            For child callback (of an `EventCallback`), this will be called
            when the event is triggered.

            :return: (bool) If the callback returns False, training is aborted early.
            """
        return True

    def _play(self, render=False):
        obs = self.eval_env.reset()
        frames = []
        done = [False] * 2
        rewards = []
        frames = []
        while not (True in done):
            actions = self.model.predictIndipendent(obs, n_envs=1, deterministic=self.deterministic)
            obs, reward, done, info = self.eval_env.step(actions.astype(np.uint8))
            rewards.append(reward)
            if render:
                frame = self.eval_env.venv.venv.vec_envs[0].par_env.env.aec_env.env.env.env.ssd_env.render(mode="RGB")
                # frames.append(im.fromarray(frame.astype(np.uint8)).resize(size=(720, 480), resample=im.BOX).convert("RGB"))
                frames.append(cv2.resize(frame, self.video_resolution, interpolation=cv2.INTER_NEAREST))
        return np.array(rewards).sum(), frames

    def _on_rollout_end(self) -> None:
        self.iterations_ += 1

        if self.iterations_ % self.render_frequency == 0:
            score, frames = self._play(render=True)
            file_name = self.logger.dir + f"/iteration_{self.iterations_+1}_score_{int(score)}.mp4"
            self.save_video(file_name, frames)

    def save_video(self, video_path, rgb_arrs, format="mp4v"):
        print("Rendering video...")
        fourcc = cv2.VideoWriter_fourcc(*format)
        video = cv2.VideoWriter(video_path, fourcc, float(15), self.video_resolution)

        for i, image in enumerate(rgb_arrs):
            video.write(image)

        video.release()

    def _on_training_end(self) -> None:
        """
        This event is triggered before exiting the `learn()` method.
        """
        pass  


class CustomIndependentCallback(BaseCallback):
    """
    A custom callback that derives from ``BaseCallback``.

    :param verbose: (int) Verbosity level 0: not output 1: info 2: debug
    """
    def __init__(self, eval_env, verbose=0, freq=1000):
        super(CustomIndependentCallback, self).__init__(verbose)
        # Those variables will be accessible in the callback
        # (they are defined in the base class)
        # The RL model
        # self.model = None  # type: BaseRLModel
        # An alias for self.model.get_env(), the environment used for training
        # self.training_env = None  # type: Union[gym.Env, VecEnv, None]
        # Number of time the callback was called
        # self.n_calls = 0  # type: int
        # self.num_timesteps = 0  # type: int
        # local and global variables
        # self.locals = None  # type: Dict[str, Any]
        # self.globals = None  # type: Dict[str, Any]
        # The logger object, used to report things in the terminal
        # self.logger = None  # type: logger.Logger
        # # Sometimes, for event callback, it is useful
        # # to have access to the parent object
        # self.parent = None  # type: Optional[BaseCallback]
        self.iterations_ = 0
        self.freq = freq
        self.eval_env = eval_env

    def set_model(self, model):
        self.model = model
        return self

    def _on_training_start(self) -> None:
        file_name = os.path.join(self.model.logger.dir, "parameters.json")

        params = {"learning_rate": self.model.learning_rate,
                  "batch_size": self.model.batch_size,
                  "gae_lambda": self.model.gae_lambda,
                  "gamma": self.model.gamma,
                  "n_envs": self.model.n_envs,
                  "n_epochs": self.model.n_epochs,
                  "normalize_advantage": self.model.normalize_advantage,
                  "target_kl": self.model.target_kl,
                  "ent_coef": self.model.ent_coef}

        json_object = json.dumps(params, indent=4)
        with open(file_name, "w") as outfile:
            outfile.write(json_object)

    def _on_rollout_start(self) -> None:
        """
        A rollout is the collection of environment interaction
        using the current policy.
        This event is triggered before collecting new samples.
        """
        pass

    def _on_step(self) -> bool:
        """
            This method will be called by the model after each call to `env.step()`.

            For child callback (of an `EventCallback`), this will be called
            when the event is triggered.

            :return: (bool) If the callback returns False, training is aborted early.
            """

        if self.n_calls % self.freq == 0:
            # env = self.model.env.venv.venv.vec_envs[0].par_env.env.aec_env.env.env.env.ssd_env
            play_env = self.eval_env
            render_env = self.eval_env.venv.venv.vec_envs[0].par_env.env.aec_env.env.env.env.ssd_env
            observations = play_env.reset()
            frames = []
            score = 0
            for _ in range(1000):
                # TODO
                actions, states = self.model.predict(observations, state=None, deterministic=False)
                observations, rewards, dones, infos = play_env.step(actions.astype(np.uint8))
                frame = render_env.render(mode="RGB")
                frames.append(im.fromarray(frame.astype(np.uint8)).resize(size=(720, 480), resample=im.BOX).convert("RGB"))
                score += rewards.sum()

            file_name = self.logger.dir + f"/iteration_{self.iterations_+1}_score_{int(score)}.gif"
            imageio.mimsave(file_name, frames, fps=15)
        return True

    def _play(self):
        env = self.eval_env
        observations = env.reset()
        frames = []
        score = 0
        for _ in range(1000):
            actions, states = self.model.predict(observations, state=None, deterministic=False)
            observations, rewards, dones, infos = env.step(actions.astype(np.uint8))

    def _on_rollout_end(self) -> None:
        self.iterations_ += 1
        # self._play()
        efficiency, equality, sustainability, peace = [], [], [], []
        ef, eq, sus, p, sor, sac = self.eval_env.par_env.env.aec_env.env.env.env.ssd_env.get_social_metrics()
        self.logger.record("metrics/efficiency", ef)
        self.logger.record("metrics/equality", eq)
        self.logger.record("metrics/sustainability", sus)
        self.logger.record("metrics/peace", p)
        self.logger.record("metrics/sum_of_rewards", sor)
        self.logger.record("metrics/shot_accuracy", sac)

    def _on_training_end(self) -> None:
        """
        This event is triggered before exiting the `learn()` method.
        """
        pass
    

class SingleAgentCallback(BaseCallback):
    """
    A custom callback that derives from ``BaseCallback``.

    :param verbose: (int) Verbosity level 0: not output 1: info 2: debug
    """
    def __init__(self, eval_env, verbose=0, render_frequency=10, deterministic=False, args={}, video_resolution=(720, 480), agent='dqn'):
        super(SingleAgentCallback, self).__init__(verbose)
        # Those variables will be accessible in the callback
        # (they are defined in the base class)
        # The RL model
        # self.model = None  # type: BaseRLModel
        # An alias for self.model.get_env(), the environment used for training
        # self.training_env = None  # type: Union[gym.Env, VecEnv, None]
        # Number of time the callback was called
        # self.n_calls = 0  # type: int
        # self.num_timesteps = 0  # type: int
        # local and global variables
        # self.locals = None  # type: Dict[str, Any]
        # self.globals = None  # type: Dict[str, Any]
        # The logger object, used to report things in the terminal
        # self.logger = None  # type: logger.Logger
        # # Sometimes, for event callback, it is useful
        # # to have access to the parent object
        # self.parent = None  # type: Optional[BaseCallback]
        self.iterations_ = 0
        self.render_frequency = render_frequency
        self.eval_env = eval_env
        self.deterministic = deterministic
        self.video_resolution = video_resolution
        self.args = args
        self.agent = agent
    def _on_training_start(self) -> None:
        file_name = os.path.join(self.model.logger.dir, "parameters.json")
        if self.agent == 'dqn':
            params = {"learning_rate": self.model.learning_rate,
                    "batch_size": self.model.batch_size,
                    'buffer_size': self.model.buffer_size,
                    'tau': self.model.tau,
                    "gamma": self.model.gamma,
                    'train_freq': self.model.train_freq.frequency,
                    'exploration_fraction': self.model.exploration_fraction,
                    "n_envs": self.model.n_envs,
                    # each frame consist from 3 channels (RGB)
                    "n_frames": self.model.env.observation_space.shape[0],
                    "policy_kwargs": self.model.policy.features_extractor_kwargs,
                    "policy_type": str(type(self.model.policy.features_extractor)).split(".")[-1],
                    "observations_space": str(self.model.observation_space)}
        elif self.agent == 'ppo':
            params = {"learning_rate": self.model.learning_rate,
                     "batch_size": self.model.batch_size,
                     "n_epochs": self.model.n_epochs,
                    #  "clip_range": self.model.clip_range,
                     "ent_coef": self.model.ent_coef,
                     "vf_coef": self.model.vf_coef,
                     "max_grad_norm": self.model.max_grad_norm,
                     "n_steps": self.model.n_steps,
                     "gamma": self.model.gamma,
                     "target_kl": self.model.target_kl,
                     "gae_lambda": self.model.gae_lambda,
                     "gamma": self.model.gamma,
                     "n_envs": self.model.n_envs,
                     "n_epochs": self.model.n_epochs,
                     "normalize_advantage": self.model.normalize_advantage,
                     "target_kl": self.model.target_kl,
                     "ent_coef": self.model.ent_coef,
                     # each frame consist from 3 channels (RGB)
                     "n_frames": self.model.env.observation_space.shape[0],
                     "policy_kwargs": self.model.policy.features_extractor_kwargs,
                     "policy_type": str(type(self.model.policy.features_extractor)).split(".")[-1],
                     "observations_space": str(self.model.observation_space)}
            
        params.update(self.args)
        
        

        json_object = json.dumps(params, indent=4)
        with open(file_name, "w") as outfile:
            outfile.write(json_object)

    def _on_rollout_start(self) -> None:
        """
        A rollout is the collection of environment interaction
        using the current policy.
        This event is triggered before collecting new samples.
        """
        pass

    def _on_step(self) -> bool:
        """
            This method will be called by the model after each call to `env.step()`.

            For child callback (of an `EventCallback`), this will be called
            when the event is triggered.

            :return: (bool) If the callback returns False, training is aborted early.
            """
        return True

    def _play(self, render=False):
        obs = self.eval_env.reset()
        frames = []
        done = [False] * 2
        rewards = []
        frames = []
        while not (True in done):
            actions, _ = self.model.predict(obs, state=None, deterministic=self.deterministic)
            obs, reward, done, info = self.eval_env.step(actions.astype(np.uint8))
            rewards.append(reward)
            if render:
                frame = self.eval_env.venv.venv.venv.vec_envs[0].par_env.env.aec_env.env.env.env.ssd_env.render(mode="RGB")
                # frames.append(im.fromarray(frame.astype(np.uint8)).resize(size=(720, 480), resample=im.BOX).convert("RGB"))
                frames.append(cv2.resize(frame, self.video_resolution, interpolation=cv2.INTER_NEAREST))
        return np.array(rewards).sum(), frames

    def _on_rollout_end(self) -> None:
        self.iterations_ += 1
        if self.iterations_ % self.render_frequency == 0:
            score, frames = self._play(render=True)
            file_name = self.logger.dir + f"/iteration_{self.iterations_+1}_score_{int(score)}.mp4"
            self.save_video(file_name, frames)

    def save_video(self, video_path, rgb_arrs, format="mp4v"):
        print("Rendering video...")
        fourcc = cv2.VideoWriter_fourcc(*format)
        video = cv2.VideoWriter(video_path, fourcc, float(15), self.video_resolution)

        for i, image in enumerate(rgb_arrs):
            video.write(image)

        video.release()

    def _on_training_end(self) -> None:
        """
        This event is triggered before exiting the `learn()` method.
        """
        pass
