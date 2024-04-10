import os
import sys
import json
import copy
import datetime
from pathlib import Path
import time
dir_path = Path(__file__).resolve().parent
sys.path.append(os.path.join(dir_path, '..'))
sys.path.append(os.path.join(dir_path, '..', 'PS-VAE', 'src'))
sys.path.append(dir_path)

import yaml
from tqdm import tqdm

import random
import numpy as np     
import torch

from summary import generate_summary
from utils.chem_utils import molecule2smiles, smiles2molecule


"""
    This file is used only to evaluate our trained policy/actor after
    training in main.py with ppo.py. I wrote this file to demonstrate
    that our trained policy exists independently of our learning algorithm,
    which resides in ppo.py. Thus, we can test our trained policy without 
    relying on ppo.py.
"""

base_path = dir_path.parent.parent

# set all seeds
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

set_seed(69)

def _log_summary(ep_len, ep_ret, ep_num):
        """
            Print to stdout what we've logged so far in the most recent episode.

            Parameters:
                None

            Return:
                None
        """
        # Round decimal places for more aesthetic logging messages
        ep_len = str(round(ep_len, 2))
        ep_ret = str(round(ep_ret, 2))

        # Print logging statements
        print(flush=True)
        print(f"-------------------- Episode #{ep_num} --------------------", flush=True)
        print(f"Episodic Length: {ep_len}", flush=True)
        print(f"Episodic Return: {ep_ret}", flush=True)
        print(f"------------------------------------------------------", flush=True)
        print(flush=True)

def rollout(policy, env_):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    config = yaml.load(open(os.path.join(base_path, 'configs/summary.yaml')), Loader=yaml.FullLoader)
    # create a copy of the environment
    env = copy.deepcopy(env_)
    if env.args.mode == 'train':
        max_timesteps_per_episode = 1
        temp = 1
        env.args.beam_size = 10


    elif env.args.mode == 'test':
        max_timesteps_per_episode = env.args.max_timesteps_per_episode
        temp = env.args.temp
    
    env.args.reward_type = 'abs'
    env.args.ratio_weight = False
    env.args.penalty = False
    env.args.ucb = False
    env.args.half_reward = False
    env.args.binary_reward = False
    env.args.bonus_for_both = False
    env.args.pred_weight = 1
    env.args.cov_weight = 10

    print(f"max_timesteps_per_episode: {max_timesteps_per_episode}")
    print(f"temp: {temp}")
    print(f"reward_type: {env.args.reward_type}, half_reward: {env.args.half_reward}, binary_reward: {env.args.binary_reward}")
    policy.to(device)


    negative_mols = env.mols['test']['negative'] # use test split
    # Open the file in write mode
    results = []
    n_candidates = {'factual': 0, 'counterfactual': 0}
    start_time = time.time()
    if env.args.most_recent:
        prefix = ''
    elif env.args.best_mean_cov:
        prefix = 'best_mean_cov'
    else:
        prefix = 'best'
    if env.args.ckpt_dir is not None:
        save_path = os.path.join(env.args.ckpt_dir, f'{env.args.mode}_{prefix}_ppo_results_infer_trained.json')
    else:
        save_path = os.path.join(env.base_path, f'{env.args.mode}_{prefix}_ppo_results_infer_trained.json')
    
    # save into results
    results_path = f"./results/{env.args.dataset}"
    if env.args.mode == 'test':
        print(f"mode: {env.args.mode}")
        print(f"eval name: {env.args.eval_name}")
        result_path = os.path.join(results_path, 'test_runs')
        if env.args.eval_name is not None:
            result_path = os.path.join(result_path, env.args.eval_name)
    else:
        result_path = os.path.dirname(save_path)
    print(f"result_path: {result_path}")

    print(env.args.ckpt_dir, save_path)
    print(f"debug mode: {env.args.debug}")
    iter_times = []
    with open(save_path, 'w') as fout:
        for idx, i in tqdm(enumerate(negative_mols), desc="Processing Molecules", total=len(negative_mols)):
            iter_start = time.time()
            # print('='*50 + 'For Mol {}'.format(i) + '='*50)
            best_rew = 0
            if not env.args.debug:
                z_vec, emb, mol = env.sample_mol('test')
            else:
                z_vec, emb, mol = env.sample_mol('train')
            ori_mol, ori_smi = mol, molecule2smiles(mol)
            obs = {'z': z_vec, 'mol': mol, 'smi': ori_smi, 'emb': emb}
            obs_ori = obs[env.args.obs_type]
            for it in range(max_timesteps_per_episode):
                done = False
                paths = []

                # Query deterministic action from policy and run it
                if obs is None:
                    continue
                obs[env.args.obs_type] = obs[env.args.obs_type].to(device)
                action = policy(obs_ori)
                if env.args.eval_mode == 'every_step':
                    obs_ori = obs[env.args.obs_type]

                # obs, rew, done, scores = env.step(obs_ori, action, mol, temp=env.args.temp)
                try:
                    obs, rew, done, scores = env.step(obs_ori, action, mol, temp=temp)
                except:
                    print('Failed to decode ---')
                    continue

                if rew > best_rew:
                    if env.args.eval_mode == 'best_reward':
                        obs_ori = obs[env.args.obs_type]
                    best_rew = rew
                    best_candidate, best_score, best_status = scores[0][0], scores[0][1], scores[0][2]
                    mol = best_candidate
                    smi = molecule2smiles(mol)
                    paths.append((mol, smi, best_score, best_status))

                    result = {ori_smi: [(x[-1], x[-2], x[-3]) for x in paths]}
                    results.append(result)

                    # Write the result to the file and flush
                    fout.write('{}\n'.format(json.dumps(result)))
                    fout.flush()
                    
                    pred = paths[0][-1]['prediction']
                    if pred >= 0.5:
                        n_candidates['counterfactual'] += 1
                    elif pred < 0.5:
                        n_candidates['factual'] += 1
                
            iter_end = time.time()
            iter_times.append(iter_end - iter_start)
            expected_secs_left = (np.mean(iter_times) * (len(negative_mols) - idx))
            expected_time_left = str(datetime.timedelta(seconds=int(expected_secs_left)))
            print(f"""
                    ----- MOLECULE {idx}/{len(negative_mols)}, (ITER {max_timesteps_per_episode})  DONE: -----
                    mean iter time: {np.mean(iter_times)}
                    expected time left: {expected_time_left}
                          ---------------------------------
                   """)

            # if env.args.debug:
            #     break

            if env.args.debug:
                # if idx > 1:
                #     break
                # wait for user to press a key to continue
                print(f"HIIII")
                input("Press Enter to continue...")
                continue
                    
        # time in h:m:s
        total_time = time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))
        print(f'Saved results to {save_path}')
        results, index_set, covered_graphs, \
        mean_coverage, mean_min_dist = generate_summary(results, 2, 
                                                        config, 
                                                        negative_mols, 
                                                        config['k'])

        print(f"result type: {type(results)}")
        results['time'] = total_time
        results['n_candidates'] = n_candidates

        if env.args.debug:
            return results, index_set, covered_graphs, mean_coverage, mean_min_dist

        res_title = f"results_{env.args.name}_{env.args.eval_mode}_{env.args.temp}_{env.args.beam_size}"
        res_name = os.path.join(result_path, res_title)
        final_results = results
        if not os.path.exists(result_path):
            os.makedirs(result_path)

        res_name = os.path.join(result_path, f"{res_title}.json")

        # if file already exists, add a number to the end of the name
        i = 1
        while os.path.exists(res_name):
            res_name = os.path.join(result_path, f"{res_title}_{i}.json")
            i += 1
        print(f"Saving results to {res_name}")
        with open(res_name, 'w') as fout:
            json.dump(final_results, fout, indent=4)
        
    return results, index_set, covered_graphs, mean_coverage, mean_min_dist



def eval_policy(policy, env):
    """
        The main function to evaluate our policy with. It will iterate a generator object
        "rollout", which will simulate each episode and return the most recent episode's
        length and return. We can then log it right after. And yes, eval_policy will run
        forever until you kill the process. 

        Parameters:
            policy - The trained policy to test, basically another name for our actor model
            env - The environment to test the policy on
            render - Whether we should render our episodes. False by default.

        Return:
            None

        NOTE: To learn more about generators, look at rollout's function description
    """

    return rollout(policy, env)