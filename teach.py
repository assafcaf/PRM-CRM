import os
import warnings
import os.path as osp

import torch
def warn(*args, **kwargs):
    pass
import warnings ; warnings.warn = lambda *args,**kwargs: None
from utils import *
warnings.warn = warn
import numpy as np
from agents import IndependentDQN, IndependentPPO, DQNRP, DQN
from stable_baselines3.common.logger import configure
import argparse
from envs import make_env
from label_schedules import LabelAnnealer, ConstantLabelSchedule
from summaries import AgentLoggerSb3
from video import SegmentVideoRecorder
from reward_models import ComparisonRewardPredictor, PrmComparisonRewardPredictor, CrmComparisonRewardPredictor

from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
from social_dilemmas.harvest_env import build_env
from social_dilemmas.callbacks.src import SingleAgentCallback
from gym.spaces import Box

warnings.filterwarnings('ignore')
CLIP_LENGTH = 1

metrics = {0: "Efficiency",
           1: "Efficiency*Peace",
           2: "Efficiency*Peace*Equality"}

def arg_pars():
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--env_id', default='harvest', type=str)  # 'PongNoFrameskip-v4'
    parser.add_argument('-n', '--name', default='syn-{}_metric-{}', type=str)
    parser.add_argument('-w', '--workers', default=24, type=int)
    parser.add_argument('-l', '--n_labels', default=int(1e4), type=int)
    parser.add_argument('-L', '--pretrain_labels', default=250, type=int)
    parser.add_argument('-t', '--num_timesteps', default=3e8, type=int)
    parser.add_argument('-a', '--agent', default="ppo", type=str, choices=["ppo", "dqn"])
    parser.add_argument('-i', '--pretrain_iters', default=500, type=int)
    parser.add_argument('-V', '--no_videos', action="store_true", default=True)
    parser.add_argument('-f', '--stacked_frames', default=1, type=int)
    parser.add_argument('-ns', '--n_steps', default=1000, type=int)
    parser.add_argument('-tf', '--train_freq', default=4, type=int)
    parser.add_argument('-na', '--n_agents', default=5, type=int)
    parser.add_argument('-pe', '--predictor_epochs', default=4, type=int)
    parser.add_argument('-ng', '--num_gpu', default="0", type=str, help="gpu id")
    parser.add_argument('-br', '--buffer_ratio', default=0.5, type=float, help='ratio of buffer size to number of labels (to reduce memory usage)')
    parser.add_argument('-d', '--debug', action="store_true", default=False)
    parser.add_argument('-c', '--same_color', action="store_true", default=False)
    parser.add_argument('-g', '--gray_scale', action="store_true", default=True)
    parser.add_argument('-ן', '--independent', action="store_true", default=True)
    parser.add_argument('-r', '--real_rewards', action="store_true", default=False)
    parser.add_argument('-sd', '--same_dim', action="store_true", default=False)
    parser.add_argument('-rp', '--rp_mode', default=1, type=int, choices=[0, 1, 2], help="reward predictor type. 0: synthetic, 1: PRM, 2: CRM")
    parser.add_argument('-m', '--metric', default=0, choices=[0, 1, 2], help="metric for RP to optimize. 0: efficiency , 1: efficiency * peace, 2: efficiency * peace * equality")
    args = parser.parse_args()
    return args


def make_reward_predictor(mode, **args):
    if mode == 0:
        return ComparisonRewardPredictor(**args)
    elif mode == 1:
        return PrmComparisonRewardPredictor(**args)
    elif mode == 2:
        return CrmComparisonRewardPredictor(**args)

def main():
    print("Setting things up...")
    
    # Parse arguments
    args = arg_pars()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.num_gpu
    pre_trian = True
    train_freq = args.train_freq * args.n_steps
    if args.debug:
        pre_trian = False
        if not args.real_rewards:
            args.workers = 2
    num_envs = args.workers   
    
    # Set up the run name and save directory 
    args.name = args.name.format(args.n_labels, metrics[args.metric])
    env_id = args.env_id
    save_dir = fr'/home/acaftory/CommonsGame/my-atari-teacher-MultiAgent3/results/{args.agent}'
    log_dirs, trail_dir = make_run_log_dirs(save_dir, env_id, args.independent, args.rp_mode, args.real_rewards, args.name, args.n_agents)


    # Set up the environment
    env = make_env(env_id, max_episode_steps=args.n_steps, same_color=args.same_color, gray_scale=args.gray_scale, same_dim=args.same_dim)  
    observation_space = Box(low=0, high=255, shape=(args.stacked_frames*env.observation_space.shape[0],) + env.observation_space.shape[1:], dtype=np.uint8)

    # Set up the logger
    sb3_loggers = [configure(osp.join(pth), ["stdout", "tensorboard"]) for pth in log_dirs]
    predictor_loggers = [AgentLoggerSb3(sb3_logger) for sb3_logger in sb3_loggers]


    # Set up the label schedule
    pretrain_labels = args.pretrain_labels if args.pretrain_labels else args.n_labels // 4
    num_timesteps = int(args.num_timesteps)
    if args.n_labels:
        label_schedules = [LabelAnnealer(
                           agent_logger=predictor_logger,
                           final_timesteps=num_timesteps,
                           final_labels=args.n_labels,
                           pretrain_labels=pretrain_labels)
                    for predictor_logger in predictor_loggers]
    else:
        print("No label limit given. We will request one label every few seconds.")
        label_schedule = ConstantLabelSchedule(pretrain_labels=pretrain_labels)

    # Set up the reward predictor
    predictor = make_reward_predictor(mode=args.rp_mode,
                                      num_agents=args.n_agents,
                                      num_envs=num_envs,
                                      agent_loggers=predictor_loggers, 
                                      label_schedules=label_schedules,
                                      fps=env.fps,
                                      observation_space=observation_space,
                                      action_space=env.action_space,
                                      stacked_frames=args.stacked_frames,
                                      device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                                      clip_length=CLIP_LENGTH,
                                      train_freq=train_freq,
                                      comparison_collector_max_len = int(args.n_labels * args.buffer_ratio),
                                      pre_train=pre_trian,
                                      epochs=args.predictor_epochs,
    )
    if pre_trian and not args.real_rewards:
        predictor.pre_trian(env_id=env_id,
                            make_env=make_env,
                            pretrain_labels=pretrain_labels,
                            clip_length=CLIP_LENGTH,
                            num_envs=num_envs,
                            n_steps=args.n_steps,
                            pretrain_iters=args.pretrain_iters,
                            same_color=args.same_color,
                            gray_scale=args.gray_scale,
                            same_dim=args.same_dim
                            )

    # Wrap the predictor to capture videos every so often:
    if not args.no_videos:
        predictor = SegmentVideoRecorder(predictor, save_dir=trail_dir)
        
    

    # build env
    print("Building environment")
    if env_id not in ['harvest']:
        vec_env = make_atari_env(env_id, n_envs=args.workers)
        vec_env = VecFrameStack(vec_env, n_stack=args.stacked_frames)
    else: 
        vec_env = build_env(rollout_len=args.n_steps, num_agents=args.n_agents, num_cpus=num_envs, same_color=args.same_color, same_dim=args.same_dim,
                            gray_scale=args.gray_scale, num_frames=args.stacked_frames, num_envs=num_envs, use_my_wrap=False, metric=args.metric)
    
    # We use a vanilla agent from openai/baselines that contains a single change that blinds it to the true reward
    # The single changed section is in `rl_teacher/agent/trpo/core.py`
    
    # building agent
    if args.agent  == "ppo":
        learner_kwargs = {'n_steps': args.n_steps,      
                         'ent_coef': 0.1,# entropy coefficient in loss
                         'batch_size': args.n_steps * args.workers // 2,  # This is from the rllib baseline implementation
                         'learning_rate':  0.0001,
                         'n_epochs': 4,
                         'gae_lambda': 0.95,
                         'gamma': 0.95,
                         'target_kl': 0.2,
                         'max_grad_norm': 40}
        agent = IndependentPPO(policy="CnnPolicy",
                               env=vec_env,
                               num_agents=args.n_agents,
                               verbose=1,
                               predictor=predictor,
                               real_rewards=args.real_rewards,
                              **learner_kwargs)
    elif args.agent  == "dqn":
        learner_kwargs = {
                          'learning_rate': 0.0005,
                          'batch_size': 128,
                          'tau': 0.01,
                          'gamma': 0.99,
                          'train_freq': 4,
                          'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                          'exploration_fraction': 0.15,
                          'learning_starts': 1e5,
                          'buffer_size': int(1e6)}
        if args.independent:
            agent = IndependentDQN(policy="CnnPolicy",
                                   env=vec_env,
                                   num_agents=args.n_agents,
                                   verbose=1,
                                   predictor=predictor,
                                   real_rewards=args.real_rewards,
                                   **learner_kwargs)
        else:
            agent = DQNRP(policy="CnnPolicy",
                          real_rewards=args.real_rewards,
                          env=vec_env,
                          verbose=1,
                          predictor=predictor,
                          **learner_kwargs)
    
    # starting agent trainig
    agent.set_logger(sb3_loggers[0])
    print("Building eval environment")
    eval_env = build_env(rollout_len=args.n_steps, num_agents=args.n_agents, num_cpus=1, same_color=args.same_color, same_dim=args.same_dim,
                        gray_scale=args.gray_scale, num_frames=args.stacked_frames, num_envs=1, use_my_wrap=False, metric=args.metric)
    callback = SingleAgentCallback(eval_env, verbose=0, render_frequency=1500, deterministic=False, args=vars(args), agent=args.agent)
    print("Starting joint training of predictor and agent")
    
    
    agent.learn(args.num_timesteps, log_interval=1, callback=callback)
    print("Training Finished")
    exit()


if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')# good solution !!!!
    main()
    
