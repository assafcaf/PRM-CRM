import os
import warnings
import os.path as osp
from time import time, sleep
import torch
import numpy as np
from agents import train_ppo, train_dqn
from agents.dqn2.agent import DQN, DQNRP
from agents.independent_dqn.agent import IndependentDQN
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
from social_dilemmas.callbacks.src import SingleDQNAgentCallback
warnings.filterwarnings('ignore')
CLIP_LENGTH = 1
metrics = {0: "Efficiency",
           1: "Efficiency*Peace",
           2: "Efficiency*Peace*Equality"}
predictors = {0: 'ComparisonRewardPredictor',
                1: 'PrmComparisonRewardPredictor',
                2: 'CrmComparisonRewardPredictor'}

def arg_pars():
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--env_id', default='harvest', type=str)  # 'PongNoFrameskip-v4'
    parser.add_argument('-n', '--name', default='syn-{}_metric-{}', type=str)
    parser.add_argument('-w', '--workers', default=16, type=int)
    parser.add_argument('-l', '--n_labels', default=int(1e4), type=int)
    parser.add_argument('-L', '--pretrain_labels', default=50, type=int)
    parser.add_argument('-t', '--num_timesteps', default=5e7, type=int)
    parser.add_argument('-a', '--agent', default="dqn", type=str)
    parser.add_argument('-i', '--pretrain_iters', default=500, type=int)
    parser.add_argument('-V', '--no_videos', action="store_true")
    parser.add_argument('-f', '--stacked_frames', default=1, type=int)
    parser.add_argument('-ns', '--n_steps', default=1000, type=int)
    parser.add_argument('-tf', '--train_freq', default=1e4, type=int)
    parser.add_argument('-na', '--n_agents', default=5, type=int)
    parser.add_argument('-ng', '--num_gpu', default="1", type=str, help="gpu id")
    parser.add_argument('-br', '--buffer_ratio', default=0.1, type=float, help='ratio of buffer size to number of labels (to reduce memory usage)')
    parser.add_argument('-d', '--debug', action="store_true", default=False)
    parser.add_argument('-c', '--same_color', action="store_true", default=True)
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

    if args.debug:
        pre_trian = False
        if not args.real_rewards:
            args.workers = 2
    num_envs = args.workers   
    
    # Set up the run name and save directory 
    args.name = args.name.format(args.n_labels, metrics[args.metric])
    env_id = args.env_id
    run_name = "%s%s/%s-%s" % (env_id,
                               f"{'-independent' if args.independent else ''}{'-' + predictors[args.rp_mode]}{'-RealRewards' if args.real_rewards else ''}",
                               args.name, int(time()))
    save_dir = fr'/home/acaftory/CommonsGame/my-atari-teacher-MultiAgent3/results/{args.agent}'


    # Set up the environment
    env = make_env(env_id, max_episode_steps=args.n_steps, same_color=args.same_color, gray_scale=args.gray_scale, same_dim=args.same_dim)  

    # Set up the logger
    sb3_logger = configure(osp.join(save_dir, run_name), ["stdout", "tensorboard"])
    predictor_logger = AgentLoggerSb3(sb3_logger)


    # Set up the label schedule
    pretrain_labels = args.pretrain_labels if args.pretrain_labels else args.n_labels // 4
    num_timesteps = int(args.num_timesteps)
    if args.n_labels:
        label_schedule = LabelAnnealer(
            agent_logger=predictor_logger,
            final_timesteps=num_timesteps,
            final_labels=args.n_labels,
            pretrain_labels=pretrain_labels)
    else:
        print("No label limit given. We will request one label every few seconds.")
        label_schedule = ConstantLabelSchedule(pretrain_labels=pretrain_labels)

    # Set up the reward predictor
    predictor = make_reward_predictor(mode=args.rp_mode,
                                      num_agents=args.n_agents,
                                      num_envs=num_envs,
                                      agent_logger=predictor_logger, 
                                      label_schedule=label_schedule,
                                      fps=env.fps,
                                      observation_space=env.observation_space,
                                      action_space=env.action_space,
                                      stacked_frames=args.stacked_frames,
                                      device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                                      clip_length=CLIP_LENGTH,
                                      train_freq=args.train_freq,
                                      comparison_collector_max_len = int(args.n_labels * args.buffer_ratio),
                                      pre_train=pre_trian
    )
    
    # ComparisonRewardPredictor(
    #     sb3_logger, 
    #     predictor_logger=predictor_logger,
    #     clip_length=CLIP_LENGTH,
    #     label_schedule=label_schedule,
    #     fps=env.fps,
    #     observation_space=env.observation_space,
    #     action_space=env.action_space,
    #     stacked_frames=args.stacked_frames,
    #     train_freq=args.train_freq,
    #     device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    #     comparison_collector_max_len = int(args.n_labels * args.buffer_ratio)
    # )
    if pre_trian:
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
        predictor = SegmentVideoRecorder(predictor, save_dir=osp.join(save_dir, run_name))
        
    

    # build env
    if env_id not in ['harvest']:
        vec_env = make_atari_env(env_id, n_envs=args.workers)
        vec_env = VecFrameStack(vec_env, n_stack=args.stacked_frames)
    else: 
        vec_env = build_env(rollout_len=args.n_steps, num_agents=args.n_agents, num_cpus=num_envs, same_color=args.same_color, same_dim=args.same_dim,
                            gray_scale=args.gray_scale, num_frames=args.stacked_frames, num_envs=num_envs, use_my_wrap=False, metric=args.metric)
    
    # We use a vanilla agent from openai/baselines that contains a single change that blinds it to the true reward
    # The single changed section is in `rl_teacher/agent/trpo/core.py`
    print("Starting joint training of predictor and agent")

    if args.agent  == "ppo":
        learner_kwargs = {'n_steps': args.n_steps,      
                    'ent_coef': 0.001,# entropy coefficient in loss
                    'batch_size': args.n_steps * args.workers // 2,  # This is from the rllib baseline implementation
                    'learning_rate':  0.0001,
                    'n_epochs': 4,
                    'gae_lambda': 0.95,
                    'gamma': 0.95,
                    'target_kl': 0.1,
                    'max_grad_norm': 40}
        train_ppo(
            vec_env=vec_env,
            predictor=predictor,
            timesteps_per_batch=args.n_steps, 
            learner_kwargs=learner_kwargs,
            sb3_logger=sb3_logger,
            num_timesteps=num_timesteps,
            use_independent_policy=args.independent_policy,
            n_agents = args.n_agents
        )
    elif args.agent  == "dqn":
        learner_kwargs = {
                          'learning_rate': 0.0005,
                          'batch_size': 128,
                          'tau': 0.02,
                          'gamma': 0.99,
                          'train_freq': 4,
                          'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                          'exploration_fraction': 0.1,
                          'learning_starts': 1e5,
                          'buffer_size': int(1e6)}
        # train_dqn(
        #     vec_env=vec_env,
        #     predictor=predictor,
        #     timesteps_per_batch=args.n_steps, 
        #     learner_kwargs=learner_kwargs,
        #     sb3_logger=sb3_logger,
        #     num_timesteps=num_timesteps,
        #     use_independent_policy=args.independent_policy
        # )
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
        agent.set_logger(sb3_logger)
        print("build eval env")
        eval_env = build_env(rollout_len=args.n_steps, num_agents=args.n_agents, num_cpus=1, same_color=args.same_color, same_dim=args.same_dim,
                            gray_scale=args.gray_scale, num_frames=args.stacked_frames, num_envs=1, use_my_wrap=False, metric=args.metric)
        
        callback = SingleDQNAgentCallback(eval_env, verbose=0, render_frequency=10000, deterministic=False, args=vars(args))
        agent.learn(int(1e8), log_interval=1, callback=callback)
        print("Training Finished")
        exit()


if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')# good solution !!!!
    main()
    
