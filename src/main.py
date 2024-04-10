"""
    This file is the executable for running PPO. It is based on this medium article: 
    https://medium.com/@eyyu/coding-ppo-from-scratch-with-pytorch-part-1-4-613dfc1b14c8
"""

# import gym
from pathlib import Path
import os
import sys
import yaml
dir_path = Path(__file__).resolve().parent

sys.path.append(str(dir_path.parent))
sys.path.append(str(dir_path.parent.parent))
sys.path.append(str((dir_path.parent.parent / 'PS-VAE' / 'src')))
sys.path.append(str(dir_path))

print(dir_path.parent)
print(dir_path.parent / 'PS-VAE' / 'src')
print(dir_path)

import torch
from arguments import get_args
from ppo import PPO
from network import FeedForwardNN, TransformerNN
from eval_policy import eval_policy

import json
from torch_geometric.data import Data, Batch, DataLoader
from generate import load_model, parallel, gen, beam_gen
from importance import Importance, GNNPredictor, mol_distance
from utils.chem_utils import molecule2smiles, smiles2molecule
from psvae_RL import PSVAEAgent
from ppo import test

import torch.nn as nn

# import os
# os.environ["CUDA_VISIBLE_DEVICES"]="7"

base_path = dir_path.parent.parent


def get_mols(dataset_name, cpus, load_weights=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # load vae
    ckpt = os.path.join(base_path, "PS-VAE/ckpts/zinc250k/constraint_prop_opt/epoch5.ckpt")
    vae_model = load_model(ckpt, -1, load_weights)
    vae_model.eval().to(device)

    MAX_POS = 100
    current_max_pos, pos_dim = vae_model.decoder.pos_embedding.weight.shape
    my_pos_embeddings = nn.Embedding(MAX_POS, pos_dim) 
    my_pos_embeddings.weight.data[:current_max_pos] = vae_model.decoder.pos_embedding.weight.data 
    my_pos_embeddings.weight.data[current_max_pos:] = vae_model.decoder.pos_embedding.weight.data[-1][None,:].repeat(MAX_POS-current_max_pos,1) 
    vae_model.decoder.pos_embedding = my_pos_embeddings 
    print(vae_model)

    splits = ['train', 'valid', 'test']
    mols_splits = {split: {} for split in splits}

    for split in splits:
        # load data
        fpath = os.path.join(base_path, f"data/{dataset_name}_{split}.smi")
        with open(fpath, 'r') as fin:
            lines = fin.read().strip().split('\n')
        sample_smis = []
        for line in lines:
            smi, logp = line.split()
            if 'Na' not in smi:
                sample_smis.append(smi)
        
        sample_smis = list(set(sample_smis))
        original_mols = parallel(smiles2molecule, sample_smis, cpus)

        mapping_info = json.load(open(os.path.join(base_path, "data/{}_mapping_info.json".format(dataset_name))))
        node_mapping = mapping_info['keep_node_mapping']
        gnn_ckpt = f'data/{dataset_name}/gnn/model_best.pth'
        predictor = GNNPredictor(gnn_ckpt, dataset_name, node_mapping, device)

        # get negative samples (p(class=0) >= 0.5)
        graphs = [predictor.convert.MolToGraph(m) for m in original_mols]
        data_loader = DataLoader(graphs, batch_size=128)
        predictions = []
        for batch in data_loader:
            preds, _ = predictor.predict(batch, target_class=0)
            predictions.extend(preds)
        negatives = [(i, m) for i, (m, p) in enumerate(zip(original_mols, predictions)) if p >= 0.5]
        _, negative_mols = zip(*negatives)
        positives = [(i, m) for i, (m, p) in enumerate(zip(original_mols, predictions)) if p < 0.5]
        _, positive_mols = zip(*positives)

        mols_splits[split]['positive'] = positive_mols
        mols_splits[split]['negative'] = negative_mols

        print(f"no. train split negative: {len(mols_splits['train']['negative'])}")

    # make train split same as test split
    if args.debug:
        mols_splits['train'] = mols_splits['test']
    
    # # save mols_splits as pkl
    # import pickle
    # save_path = "data/aids/split_data/"
    # print(f"save path: {save_path}")
    # with open(os.path.join(save_path, f"{dataset_name}_mols_splits.pkl"), 'wb') as fout:
    #     print(f"Saving mols_splits to {os.path.join(base_path, f'data/{dataset_name}_mols_splits.pkl')}")
    #     pickle.dump(mols_splits, fout)
    #     exit()

    return vae_model, predictor, mols_splits



def train(env, hyperparameters, actor_model, critic_model, args, **kwargs):
    """
        Trains the model.

        Parameters:
            env - the environment to train on
            hyperparameters - a dict of hyperparameters to use, defined in main
            actor_model - the actor model to load in if we want to continue training
            critic_model - the critic model to load in if we want to continue training

        Return:
            None
    """	
    print(f"Training", flush=True)

    # Create a model for PPO.
    # TransformerNN, FeedForwardNN
    # Combine hyperparameters and kwargs
    model = PPO(policy_class=TransformerNN, env=env, args=args, **hyperparameters)

    # Tries to load in an existing actor/critic model to continue training on
    if actor_model != None and critic_model != None:
        print(f"Loading in {actor_model} and {critic_model}...", flush=True)
        model.actor.load_state_dict(torch.load(actor_model))
        model.critic.load_state_dict(torch.load(critic_model))
        print(f"Successfully loaded.", flush=True)
    elif actor_model != None or critic_model != None: # Don't train from scratch if user accidentally forgets actor/critic model
        print(f"Error: Either specify both actor/critic models or none at all. We don't want to accidentally override anything!")
        sys.exit(0)
    else:
        print(f"Training from scratch.", flush=True)

    # Train the PPO model with a specified total timesteps
    # NOTE: You can change the total timesteps here, I put a big number just because
    # you can kill the process whenever you feel like PPO is converging
    model.learn(total_timesteps=args.total_timesteps)

def main(args):
    """
        The main function to run.

        Parameters:
            args - the arguments parsed from command line

        Return:
            None
    """
    # NOTE: Here's where you can set hyperparameters for PPO. I don't include them as part of
    # ArgumentParser because it's too annoying to type them every time at command line. Instead, you can change them here.
    # To see a list of hyperparameters, look in ppo.py at function _init_hyperparameters
    if args.ckpt_dir is not None:
        args.actor_model = os.path.join(args.ckpt_dir, 'best_ppo_actor.pt')
        args.critic_model = os.path.join(args.ckpt_dir, 'best_ppo_critic.pt')
        args.config = os.path.join(args.ckpt_dir, 'config.yml')
        print(f"Loading from {args.actor_model} and {args.critic_model}...", flush=True)

    config_args = None if args.config is None else yaml.load(open(args.config), Loader=yaml.FullLoader)
    if config_args is not None:
        for attr, value in config_args.items():
            if attr not in ['mode', 'wandb', 
                            'max_timesteps_per_episode',
                            'ckpt_dir', 'debug', 'temp',
                            'beam_size', 'eval_name', 'batched', 
                            'most_recent', 'best_mean_cov', 
                            'rand_actor', 'no_actor']:
                print(f"Setting {attr} to {value}")
                setattr(args, attr, value)
    
    
    if args.mode == 'test':
        print(f"Setting batch size to 1 for testing.", flush=True)
        args.wandb = False
        # args.max_timesteps_per_episode = 20
        # args.temp = 1
        # args.reward_type = 'abs'
    if args.debug == True:
        # args.max_timesteps_per_episode = 1
        # args.beam_size = 1
        # args.cov_weight = 25
        # args.ratio_weight = True
        args.batch_size = 3
        pass

    assert args.dataset is not None, "Must specify dataset"
    
    hyperparameters = {
                'timesteps_per_batch': args.batch_size, 
                'max_timesteps_per_episode': args.max_timesteps_per_episode, 
                'gamma': 0.99, 
                'n_updates_per_iteration': args.n_updates_per_iteration, # TODO: edddie: look at these
                'lr': args.lr, # TODO: edddie: play round
                'clip': 0.2, # TODO: edddie: look at these
              }

    # Creates the environment we'll be running. If you want to replace with your own
    # custom environment, note that it must inherit Gym and have both continuous
    # observation and action spaces.
    
    print(f"rand_psvae: {args.rand_psvae}")
    psvae, predictor, mols = get_mols(args.dataset, 32, not args.rand_psvae)
    env = PSVAEAgent(psvae, predictor, mols, args)

    # Train or test, depending on the mode specified
    if args.mode == 'train':
        train(env=env, 
              hyperparameters=hyperparameters, 
              actor_model=args.actor_model, 
              critic_model=args.critic_model,
              args=args)
    else:
        model = PPO(policy_class=TransformerNN, env=env, args=args, **hyperparameters)
        if isinstance(args.actor_model, str):
            model.actor.load_state_dict(torch.load(args.actor_model, map_location=torch.device('cpu')))
        else:
            print("-------- No actor model specified for testing. -------")
        test(env=env, actor_model=model.actor)

# if __name__ == '__main__':
# 	args = get_args() # Parse arguments from command line
# 	print(f"Mode: {args.mode}", flush=True)
# 	main(args)


import cProfile
import pstats
import time
import multiprocessing as mp

if __name__ == '__main__':
    args = get_args() 
    print(f"args: {args}")
    print(f"Mode: {args.mode}", flush=True)

    if args.n_envs > 1:
        mp.set_start_method('spawn')

    if args.debug:
        # Create a profiler
        profiler = cProfile.Profile()

        # Run your main function under the profiler
        start_time = time.time()
        profiler.runcall(main, args)
        end_time = time.time()

        # Print total time taken
        print(f"Total time taken: {end_time - start_time} seconds")

        # Create a Stats object from the profiler
        stats = pstats.Stats(profiler)

        # Sort by cumulative time and print stats for the top 20 functions
        stats.sort_stats(pstats.SortKey.CUMULATIVE).print_stats(20)

        # (Optional) Filter to show only stats for functions in a specific module (e.g., pandas)
        # stats.print_stats('pandas')

        # # Dump the stats to a file for further analysis
        # stats.dump_stats("profile_stats_batched_parallel_2.prof")

        # Optionally, you can open this file with a tool like SnakeViz for more in-depth analysis:
        # snakeviz profile_stats.prof (run this command in your terminal)
    else:
        main(args)