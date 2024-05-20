from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import obs_as_tensor
import numpy as np
import os
import json
import cv2


class IndependentAgentCallback(BaseCallback):
    """
    A custom callback that derives from ``BaseCallback``.

    :param verbose: (int) Verbosity level 0: not output 1: info 2: debug
    """
    def __init__(self, eval_env, verbose=0, freq=1000):
        super(IndependentAgentCallback, self).__init__(verbose)
        self.iterations_ = 0
        self.freq = freq
        self.eval_env = eval_env

    def _on_training_start(self) -> None:
        file_name = os.path.join(self.model.logger.dir, "parameters.json")

        params = {"learning_rate": self.model.learning_rate,
                  "batch_size": self.model.agents[0].batch_size,
                  "gae_lambda": self.model.gae_lambda,
                  "gamma": self.model.gamma,
                  "n_envs": self.model.n_envs,
                  "n_epochs": self.model.agents[0].n_epochs,
                  "normalize_advantage": self.model.agents[0].normalize_advantage,
                  "target_kl": self.model.agents[0].target_kl,
                  "ent_coef": self.model.ent_coef,
                  "n_frames": self.model.env.observation_space.shape[-1],
                  "policy_type": str(type(self.model.agents[0].policy.features_extractor)).split(".")[-1],
                  "observations_space": str(self.model.observation_space),
                  "actions": str(self.eval_env.venv.venv.vec_envs[0].par_env.env.aec_env.env.env.env.ssd_env.spawn_prob)
                  }

        json_object = json.dumps(params, indent=4)
        with open(file_name, "w") as outfile:
            outfile.write(json_object)

    def _on_step(self) -> bool:
        pass

    def _on_rollout_end(self) -> None:
        if (self.iterations_ + 1) % self.freq == 0:
            play_env = self.eval_env
            render_env = self.eval_env.venv.venv.vec_envs[0].par_env.env.aec_env.env.env.env.ssd_env
            observations = play_env.reset()
            frames = []
            score = 0
            for _ in range(1000):
                values, log_probs, clipped_actions, observations_, clipped_actions = [], [], [], [], []
                actions = []
                self.model.predict_low_level(observations, actions, values, log_probs, clipped_actions, observations_,
                                  self.model.num_agents, 1, self.model.agents_observation_space)
                clipped_actions.append(np.zeros(1, dtype=np.int64))
                all_clipped_actions = np.vstack(clipped_actions).transpose().reshape(-1)
                observations, rewards, dones, infos = play_env.step(all_clipped_actions)
                frames.append(render_env.render())
                score += rewards.sum()

            file_name = f"/iteration_{self.iterations_ + 1}_score_{int(score)}"
            self.make_video_from_rgb_imgs(frames, vid_path= self.logger.dir, video_name=file_name)
        self.iterations_ += 1

    @staticmethod
    def make_video_from_rgb_imgs(rgb_arrs, vid_path, video_name='trajectory',
                                 fps=15, format="mp4v", resize=(720, 480)):
        """
        Create a video from a list of rgb arrays
        """
        print("Rendering video...")
        if vid_path[-1] != '/':
            vid_path += '/'
        video_path = vid_path + video_name + '.mp4'

        if resize is not None:
            width, height = resize
        else:
            frame = rgb_arrs[0]
            height, width, layers = frame.shape

        fourcc = cv2.VideoWriter_fourcc(*format)
        video = cv2.VideoWriter(video_path, fourcc, float(fps), (width, height))

        for i, image in enumerate(rgb_arrs):
            if resize is not None:
                image = cv2.resize(image, resize, interpolation=cv2.INTER_NEAREST)
            video.write(image)

        video.release()
        cv2.destroyAllWindows()