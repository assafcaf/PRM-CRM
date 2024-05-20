from agents.dqn.agents import DQN #, IndependentDQN
import numpy as np
from agents.dqn.rollouts import ParallelRollout


def print_stats(stats):
    for k, v in stats.items():
        if 'time' in k.lower():
            minutes = int(v / 60)
            if minutes:
                v = "{:02d}:{:04.1f}   ".format(minutes, v - minutes * 60)
            else:
                v = "   {:04.2f}  ".format(v)
        elif isinstance(v, int):
            v = str(v) + "     "
        elif isinstance(v, (float, np.floating)):
            v = "{:.4f}".format(v)

        print("{:38} {:>12}".format(k + ":", v))

def train_dqn(
        vec_env,
        predictor,
        sb3_logger=None,
        timesteps_per_batch=5000,
        seed=0,
        learner_kwargs={},
        num_timesteps=int(1e6),
        use_independent_policy=False,
):
    
    # vuild sb3 learner
    if not use_independent_policy:
        learner =  DQN("MlpPolicy",
                        vec_env,                  
                        verbose=1,
                        **learner_kwargs)
        rollouts = ParallelRollout(vec_env, predictor, learner, seed)
    # else:
    #     learner = IndependentDQN(n_agents = n_agents,
    #                              policy="MlpPolicy",
    #                              env=vec_env,                  
    #                              verbose=1,
    #                              **learner_kwargs)
    #     rollouts = ParallelRolloutIndepentent(vec_env, predictor, learner, seed)
    learner.set_logger(sb3_logger)
    learner.init_training(num_timesteps)
    
    # buid rollouts for predictor and learner



    t = 0
    while t < num_timesteps:

        # collect rollouts
        paths = rollouts.rollout(timesteps_per_batch)

        # train reward predictor and learner
        for path in paths:
            predictor.path_callback(path)
        learner.train(batch_size=learner.batch_size, gradient_steps=learner.gradient_steps)
        learner.after_train()
        t += vec_env.num_envs * timesteps_per_batch
        print('num_timesteps', t, 'max_timesteps', num_timesteps)
    print("Training finished?")