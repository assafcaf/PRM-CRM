
from social_dilemmas.harvest_env import build_env
from social_dilemmas.callbacks.src import SingleAgentCallback
from stable_baselines3 import PPO


def main():
    n_steps = 200
    tensorboard_log = r"/home/acaftory/CommonsGame/my-atari-teacher-MultiAgent2/results/harvestNoRp"



    num_envs = 8  # number of parallel multi-agent environments
    num_frames = 6  # number of frames to stack together; use >4 to avoid automatic VecTransposeImage
    ent_coef = 0.01 # entropy coefficient in loss
    batch_size = n_steps * num_envs // 2  # This is from the rllib baseline implementation
    lr = 0.0001
    n_epochs = 4
    gae_lambda = 0.95
    gamma = 0.99
    target_kl = 0.2
    grad_clip = 40
    verbose = 3
    num_agents = 1

    vec_env = build_env(rollout_len=n_steps, num_agents=num_agents, num_cpus=num_envs,
                        num_frames=num_frames, num_envs=num_envs, use_my_wrap=False)
    
    eval_env = build_env(rollout_len=n_steps, num_agents=num_agents, num_cpus=1,
                    num_frames=num_frames, num_envs=1, use_my_wrap=False)
    
    model = PPO(
        "MlpPolicy",
        env=vec_env,
        learning_rate=lr,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        gae_lambda=gae_lambda,
        ent_coef=ent_coef,
        max_grad_norm=grad_clip,
        target_kl=target_kl,
        tensorboard_log=tensorboard_log,
        verbose=verbose,
    )
    callback = SingleAgentCallback(render_frequency=10, eval_env=eval_env)
    model.learn(total_timesteps=10000000, callback=callback)


main()
