"""
    This file contains the arguments to parse at command line.
    File main.py will call get_args, which then the arguments
    will be returned.
"""
import argparse

def get_args():
    """
        Description:
        Parses arguments at command line.

        Parameters:
            None

        Return:
            args - the arguments parsed
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default=None, type=str, help='Dataset name')
    parser.add_argument('--mode', dest='mode', type=str, default='train')              # can be 'train' or 'test'
    parser.add_argument('--actor_model', dest='actor_model', type=str, default=None)     # your actor model filename
    parser.add_argument('--critic_model', dest='critic_model', type=str, default=None)   # your critic model filename
    parser.add_argument('--total_timesteps', dest='total_timesteps', type=int, default=1e6)  # total timesteps to train
    parser.add_argument('--wandb', action='store_true', default=False, help='Use wandb for logging')
    parser.add_argument('--name', type=str, default='ppo', help='Name of the run')
    parser.add_argument('--run_id', type=str, default=None, help='Run ID')
    parser.add_argument('--eval_every', type=int, default=10, help='Evaluate every x episodes')
    parser.add_argument('--debug', action='store_true', default=False, help='Debug mode')
    parser.add_argument('--ckpt_dir', type=str, default=None, help='Checkpoint directory')
    parser.add_argument('--most_recent', action='store_true', default=False, help='Use most recent checkpoint')
    parser.add_argument('--best_mean_cov', action='store_true', default=False, help='Use best mean coverage checkpoint')
    parser.add_argument('--rand_actor', action='store_true', default=False, help='Random actor for ablations')
    parser.add_argument('--rand_psvae', action='store_true', default=False, help='Random psvae for ablations')
    parser.add_argument('--eval_mode', type=str, default='every_step', choices=['every_step', 'best_reward', 'no_update'], help='Evaluation mode')
    parser.add_argument('--eval_name', type=str, default=None, help='Name of the run')
    parser.add_argument('--config', type=str, default=None, help='Config file')
    parser.add_argument('--n_envs', type=int, default=1, help='Number of parallel environments')
    parser.add_argument('--batched', action='store_true', default=False, help='Batched decoding')
    parser.add_argument('--actor_type', type=str, default='psvae', choices=['psvae', 'transformer', 'no_actor'], help='Type of actor network')
    parser.add_argument('--obs_type', type=str, choices=['emb', 'z'], default='graph', help='Observation type')
    parser.add_argument('--action_type', type=str, choices=['mu', 'sigma', 'z', 'psvae', 'mapping'], default='mu', help='mu, sigma, or z')
    parser.add_argument('--max_timesteps_per_episode', type=int, default=1, help='Max timesteps per episode')
    parser.add_argument('--beam_size', type=int, default=10, help='Number of beams for beam search')
    parser.add_argument('--temp', type=float, default=1.0, help='Temperature for beam decoding')
    parser.add_argument('--entropy_reg', type=float, default=None, help='Entropy regularization')
    parser.add_argument('--sgd', action='store_true', default=False, help='Use SGD instead of Adam')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--lr_scheduler', action='store_true', default=False, help='Use learning rate scheduler')
    parser.add_argument('--n_updates_per_iteration', type=int, default=1, help='Number of updates per iteration')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--cov_weight', type=float, default=10, help='Coverage weight')
    parser.add_argument('--pred_weight', type=float, default=1, help='Prediction weight')
    parser.add_argument('--ratio_weight', action='store_true', default=False, help='Use ratio weight')
    parser.add_argument('--binary_reward', action='store_true', default=False, help='Use binary reward')
    parser.add_argument('--half_reward', action='store_true', default=False, help='Use half reward')
    parser.add_argument('--bonus_for_both', action='store_true', default=False, help='Bonus for both')
    parser.add_argument('--penalty', type=float, default=0, help='Penalty for not satisfying constraints')
    parser.add_argument('--ucb', action='store_true', default=False, help='Use UCB for samlping')
    parser.add_argument('--reward_type', type=str, default='reward_plus_delta', choices=['abs', 'reward_plus_delta', 'delta'], help='Reward type')
    args = parser.parse_args()

    return args
