import argparse
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--env_id', default='Breakout-v0')
    parser.add_argument('-p', '--reward_model', default='synth', type=str)
    parser.add_argument('-n', '--name', default='synth-test', type=str)
    parser.add_argument('-s', '--seed', default=1, type=int)
    parser.add_argument('-w', '--workers', default=4, type=int)
    parser.add_argument('-l', '--n_labels', default=300, type=int)
    parser.add_argument('-L', '--pretrain_labels', default=None, type=int)
    parser.add_argument('-t', '--num_timesteps', default=5e6, type=int)
    parser.add_argument('-a', '--agent', default="ga3c", type=str)
    parser.add_argument('-i', '--pretrain_iters', default=10, type=int)
    parser.add_argument('-b', '--starting_beta', default=0.1, type=float)
    parser.add_argument('-c', '--clip_length', default=1.5, type=float)
    parser.add_argument('-f', '--stacked_frames', default=4, type=int)
    parser.add_argument('-V', '--no_videos', action="store_true")
    parser.add_argument('--force_new_environment_clips', action="store_true")
    parser.add_argument('--force_new_training_labels', action="store_true")
    parser.add_argument('--force_new_reward_model', action="store_true")
    parser.add_argument('--force_new_agent_model', action="store_true")
    args = parser.parse_args()
    
    return args