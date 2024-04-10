"""
    The file contains the PPO class to train with.
    NOTE: All "ALG STEP"s are following the numbers from the original PPO pseudocode.
            It can be found here: https://spinningup.openai.com/en/latest/_images/math/e62a8971472597f4b014c2da064f636ffe365ba3.svg
"""

from pathlib import Path
import os
import sys
dir_path = Path(__file__).resolve().parent
sys.path.append(str(dir_path.parent))
sys.path.append(str(dir_path.parent.parent))
sys.path.append(str((dir_path.parent.parent / 'PS-VAE' / 'src')))
sys.path.append(str(dir_path))

import numpy as np
import yaml
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.distributions import MultivariateNormal

import pytorch_lightning as pl
from pl_models.ps_vae_model import PSVAEModel
from utils.chem_utils import molecule2smiles, smiles2molecule
from rdkit import Chem

from importance import Importance, GNNPredictor, mol_distance
from data.bpe_dataset import BPEMolDataset, get_dataloader
from generate import parallel, gen, gen_batch, beam_gen, load_model

import time

base_path = dir_path.parent.parent


configs = {'threshold': 0.87,
           'beam_size': 10, 
           'max_atom_num': 60,
           'add_edge_th': 0.5,
           'obs_dim': "embedding"}

# config_path = "configs/config.yaml"
# configs = yaml.load(open(config_path, 'r'), Loader=yaml.FullLoader)

def beam_gen_2(model, z, beam, max_atom_num, add_edge_th, temperature, constraint_mol=None):
    gens = [gen(model, z, max_atom_num, add_edge_th, temperature, constraint_mol) for _ in range(beam)]
    mols = [model.return_data_to_mol(g) for g in gens]
    return mols

def beam_gen_batch(model, z, beam, max_atom_num, add_edge_th, temperature, constraint_mol=None):
    gens = gen_batch(model, z, beam, max_atom_num, add_edge_th, temperature, constraint_mol)
    mols = [model.return_data_to_mol(g) for g in gens]
    return mols

from multiprocessing import Pool

def worker(args):
    model, z, max_atom_num, add_edge_th, temperature, constraint_mol = args
    return gen(model, z, max_atom_num, add_edge_th, temperature, constraint_mol)

def beam_gen_2_parr(model, z, beam, max_atom_num, add_edge_th, temperature, constraint_mol=None):
    with Pool() as p:
        gens = p.map(worker, [(model, z, max_atom_num, add_edge_th, temperature, constraint_mol) for _ in range(beam)])
    mols = [model.return_data_to_mol(g) for g in gens]
    return mols

def softmax(x):
    e_x = np.exp(x - np.max(x))
    e_x = np.where(np.isnan(e_x), 1e-10, e_x)  # replace NaNs with a small number
    return e_x / e_x.sum()

def update_config(configs, args):
    configs['beam_size'] = args.beam_size
    configs['action_type'] = args.action_type
    configs['obs_type'] = args.obs_type
    configs['obs_dim'] = 56 if args.obs_type == 'z' else 400
    configs['act_dim'] = 56 if args.action_type in ['z', 'psvae'] else 400
    if args.action_type in ['z', 'mapping']:
        configs['act_dim'] = 56
    else:
        configs['act_dim'] = 400
    return configs

class PSVAEAgent(nn.Module):
    def __init__(self, ps_vae, predictor, mols, args):
        super().__init__()
        self.config = update_config(configs, args)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.ps_vae = ps_vae.to(self.device)
        self.mols = mols
        self.scorer = Importance(mols['train']['negative'], 
                                 predictor, 
                                 threshold=configs['threshold'], 
                                 norm=False)
        self.predictor = predictor
        self.obs_dim = configs['obs_dim']
        self.act_dim = configs['act_dim']
        self.args = args
        # make sure everything is on the same device

        # episode buffer
        self.ep_obs = []
        self.ep_acts = []
        self.ep_rewds = []
    
    def duplicate(self):
        return PSVAEAgent(self.ps_vae, self.predictor, self.mols, self.args)

    def reset(self):
        self.ep_obs = []
        self.ep_acts = []
        self.ep_rewds = []
    
    def calculate_ucb_values(self, mols):
        total_samples = sum(self.mol_rewards[mol]['count'] for mol in mols if mol in self.mol_rewards)
        ucb_values = []

        for mol in mols:
            mol = Chem.MolToSmiles(mol)
            if mol in self.mol_rewards and self.mol_rewards[mol]['count'] > 0:
                count = self.mol_rewards[mol]['count']
                total_reward = self.mol_rewards[mol]['total_reward']
                avg_reward = total_reward / count
                ucb = avg_reward + np.sqrt(2 * np.log(total_samples) / count)
            else:
                # print(f"Count for molecule {mol} is 0")
                ucb = float('inf')
            ucb_values.append(ucb)
        
        max_ucb = max(val if val != float('inf') else 0 for val in ucb_values)
        adjusted_ucb = [val if val != float('inf') else max_ucb + 1 for val in ucb_values]
        print("------ N UCB NON-INF VALS: -------", sum(ucb != float('inf') for ucb in ucb_values))
        return adjusted_ucb

    def sample_mol(self, split, use_ucb=False):
        """
        UCB Sampling: Sample molecules according to
        total reward and variance.
        """
        mols = self.mols[split]['negative']

        if use_ucb:
            print('...... using UCB ......')
            ucb_values = self.calculate_ucb_values(mols)
            probabilities = softmax(ucb_values)
            index = np.random.choice(range(len(mols)), p=probabilities)
        else:
            # Random sampling
            index = np.random.randint(len(mols))

        mol = mols[index]
        while mol.GetNumAtoms() <= 5 or mol.GetNumBonds() <= 6 or mol.GetNumAtoms() >= 100:
            if use_ucb:
                ucb_values[index] = -float('inf')  # Exclude this molecule from further consideration
                index = np.argmax(ucb_values)
            else:
                index = np.random.randint(len(mols))
            mol = mols[index]

        # mol = Chem.RemoveHs(mol)
        z_vecs, graph_embedding = self.get_z_from_mol(Chem.RemoveAllHs(mol))
        return z_vecs.squeeze(), graph_embedding.squeeze(), mol

    def get_obs_space(self, type):
        if type == 'z':
            return 56
        elif type == 'embedding':
            return 400
        else:
            raise NotImplementedError

    def get_z_(self, batch):
        x, edge_index, edge_attr = batch['x'], batch['edge_index'], batch['edge_attr']
        x_pieces, x_pos = batch['x_pieces'], batch['x_pos']
        x = self.ps_vae.decoder.embed_atom(x, x_pieces, x_pos)
        batch_size, node_num, node_dim = x.shape
        graph_ids = torch.repeat_interleave(torch.arange(0, batch_size, device=x.device), node_num)
        _, all_x = self.ps_vae.encoder.embed_node(x.view(-1, node_dim), edge_index, edge_attr)
        # [batch_size, dim_graph_feature]
        graph_embedding = self.ps_vae.encoder.embed_graph(all_x, graph_ids, batch['atom_mask'].flatten())
        z_vecs, _ = self.ps_vae.decoder.rsample(graph_embedding)
        return z_vecs, graph_embedding

    def get_sigma(self, conds):
        batch_size = conds.shape[0]
        z_log_var = -torch.abs(self.ps_vae.decoder.W_log_var(conds)) #Following Mueller et al.
        z_var = torch.exp(z_log_var / 2)
        return z_var
    
    def get_res(self, gen):
        '''reprocess generated data and encode its latent variable
        gen is an instance of Mol of rdkit.Chem'''
        step1_res = BPEMolDataset.process_step1(gen, self.ps_vae.tokenizer)
        step2_res = BPEMolDataset.process_step2(step1_res, self.ps_vae.tokenizer)
        res = BPEMolDataset.process_step3([step2_res], self.ps_vae.tokenizer, device=self.ps_vae.device)
        return res
            
    def get_z_from_mol(self, mol):
        res = self.get_res(mol)
        z_vecs, graph_embedding = self.get_z_(res)
        return z_vecs, graph_embedding

    def get_new_z(self, conds, action, action_var=None):
        batch_size = conds.shape[0]
        # add the new mean to the action
        # TODO: also change the variance?, perturb embeddings vs. z
        if self.config['action_type'] == 'psvae':
            return conds, action
        elif self.config['obs_type'] == 'emb':
            # if emb then calc new z
            z_mean = self.ps_vae.decoder.W_mean(conds)
            if self.config['action_type'] in ['mu', 'sigma']:
                z_mean += action
            z_log_var = -torch.abs(self.ps_vae.decoder.W_log_var(conds)) #Following Mueller et al.
            # kl_loss = -0.5 * torch.sum(1.0 + z_log_var - z_mean * z_mean - torch.exp(z_log_var)) / batch_size
            epsilon = torch.randn_like(z_mean)
            z_var = torch.exp(z_log_var / 2)
            if self.config['action_type'] == 'sigma':
                z_var += action_var
            z_vecs = z_mean + z_var * epsilon
        elif self.config['action_type'] == 'no_actor':
            # else conds is already z
            z_vecs = conds
            z_mean = None
            exit()
        if self.config['action_type'] == 'z':
            # print('z_vecs, action', z_vecs.shape, action.shape)
            z_vecs += action
        elif self.config['action_type'] == 'mapping':
            z_vecs = action
            z_mean = None
        return z_mean, z_vecs

    def step(self, conds, action, ori_mol, temp=None):
        """
        -- 1 episode --
        TODO
        1. generate 1 molecule ✅
        2. generate 10 molecules ✅
            - calculate the average reward for 10 molecules ✅

        -- temperature (beam_gen_2) --
        1. keep temperature stable
        2. decrease temperature according to reward?
        """
        z_mean, z_vecs = self.get_new_z(conds, action)

        if not self.args.batched:
            mols = beam_gen_2(self.ps_vae, z_vecs, 
                              configs['beam_size'], configs['max_atom_num'], configs['add_edge_th'], 
                              self.args.temp if temp is None else temp, 
                              constraint_mol=ori_mol)
        else:
            # print(f"--- z_vecs shape: {z_vecs.shape}, conds shape: {conds.shape} --- ")
            # print(f"cons mol type: {type(ori_mol)}")
            # exit()
            mols = beam_gen_batch(self.ps_vae, z_vecs, 
                                   configs['beam_size'], configs['max_atom_num'], configs['add_edge_th'], 
                                   self.args.temp if temp is None else temp, 
                                   constraint_mol=ori_mol)
             
        # mols = beam_gen_2_parr(self.ps_vae, z_vecs.detach(), configs['beam_size'], configs['max_atom_num'], configs['add_edge_th'], self.args.temp, constraint_mol=ori_mol)
        mols = [m for m in mols if m is not None]
        if len(mols) == 0:
            print("No valid molecules generated")
            return None, 0, None, None
        
        rewards = []
        scores = []

        # Pass the entire list of molecules to get_importance
        scores_, status_ = self.scorer.get_importance_batch(mols,
                                                            batch_size=len(mols),
                                                            target_class=1, 
                                                            keys=['coverage', 'prediction'],
                                                            weights=[self.args.cov_weight, 
                                                            self.args.pred_weight],
                                                            penalty=self.args.penalty,
                                                            debug=self.args.debug,
                                                            binary_reward=self.args.binary_reward,
                                                            ratio_weight=self.args.ratio_weight,
                                                            half_reward=self.args.half_reward,
                                                            bonus_for_both=self.args.bonus_for_both)
        # Append molecule score to the buffer
        self.ep_rewds.append(scores_)

        # Convert scores to a list if it's a tensor
        if isinstance(scores_, torch.Tensor):
            scores_ = scores_.tolist()

        # Convert each molecule to SMILES and append to the scores list
        for mol, score, status in zip(mols, scores_, status_):
            smi = molecule2smiles(mol)
            scores.append((mol, score, status))
            rewards.append(score)
        mean_reward = np.mean(rewards)
        print(f"mean reward: {mean_reward}")
        
        # if this is not the first step, compare to previous step
        if len(self.ep_rewds) > 1:
            score_prev = self.ep_rewds[-2]
        else:
            score_prev, _, = self.scorer.get_importance_batch([ori_mol], 
                                                                target_class=1, 
                                                                keys=['coverage', 'prediction'], 
                                                                weights=[self.args.cov_weight,
                                                                self.args.pred_weight],
                                                                penalty=self.args.penalty,                                                                        
                                                                debug=self.args.debug,
                                                                binary_reward=self.args.binary_reward,
                                                                ratio_weight=self.args.ratio_weight,
                                                                half_reward=self.args.half_reward,
                                                                bonus_for_both=self.args.bonus_for_both)

        print(f"score_prev: {score_prev}, mean_reward: {mean_reward}")
        
        reward_delta = np.mean(mean_reward - score_prev)
        print(f"reward_delta: {mean_reward} - {score_prev} = {reward_delta}, reward: {mean_reward}")
        # wether to calculate delta or absolute (for eval) reward
        if self.args.reward_type == 'delta':
            mean_reward = reward_delta
        elif self.args.reward_type == 'abs':
            mean_reward = mean_reward
        elif self.args.reward_type == 'reward_plus_delta':
            mean_reward = mean_reward + reward_delta
        else:
            raise NotImplementedError(f"reward type {self.args.reward_type} not implemented")
        
        # if max number of steps reached, return done=True flag
        if (len(self.ep_rewds) >= self.args.max_timesteps_per_episode or 
            (self.args.reward_type == 'delta' and reward_delta < 0)):
            status = True
        else:
            status = False

        # results sorted by best candiate
        scores = sorted(scores, key=lambda x: x[1], reverse=True)

        # our final observation is made up of the a) output molecule b) new z vector  
        # mol = mols[0]
        mol = [m for m in mols if m.GetNumAtoms() > 1 and m.GetNumBonds() > 1][0]
        if self.args.obs_type == 'z':
            emb = None
        if self.args.obs_type == 'emb':
            if self.args.action_type != 'psvae':
                # z, emb = self.get_res(mol)
                print('mol', mol.GetNumAtoms(), mol.GetNumBonds())
                z, emb = self.get_z_from_mol(mol)
            elif self.args.action_type == 'psvae':
                z, emb = z_vecs, conds
        
        obs = {'mol': mol, 'emb': emb, 'z': z_vecs}
        return obs, mean_reward, status, scores

        
    # def step(self, conds, action, ori_mol, curr_temp=0.1):
    #     ## get new z
    #     batch_size = conds.shape[0]
    #     # add the new mean to the action
    #     # TODO: also change the variance?, perturb embeddings vs. z
    #     z_mean = self.ps_vae.decoder.W_mean(conds) + action
    #     z_log_var = -torch.abs(self.ps_vae.decoder.W_log_var(conds)) #Following Mueller et al.
    #     # kl_loss = -0.5 * torch.sum(1.0 + z_log_var - z_mean * z_mean - torch.exp(z_log_var)) / batch_size
    #     epsilon = torch.randn_like(z_mean)
    #     z_var = torch.exp(z_log_var / 2)
    #     z_vecs = z_mean + z_var * epsilon

    #     ## get new reward
    #     """
    #     -- 1 episode --
    #     TODO
    #     1. generate 1 molecule ✅
    #     2. generate 10 molecules ✅
    #         - calculate the average reward for 10 molecules ✅

    #     -- temperature (beam_gen_2) --
    #     1. keep temperature stable
    #     2. decrease temperature according to reward?
    #     """
    #     mols = beam_gen_2(self.ps_vae, z_vecs, configs['beam_size'], configs['max_atom_num'], configs['add_edge_th'], curr_temp, constraint_mol=ori_mol)
    #     rewards = []
    #     scores = []
    #     for mol in mols:
    #         score, status, details, = self.scorer.get_importance(mol, target_class=1, keys=['coverage', 'prediction'], weights=[1, 1])
    #         rewards.append(score)
    #         smi = molecule2smiles(mol)
    #         scores.append((mol, score, status))
        
    #     mean_reward = np.mean(rewards)
    #     # results sorted by best candiate
    #     scores = sorted(scores, key=lambda x: x[1], reverse=True)
            
    #     return z_mean, mean_reward, status, scores

class PSVAEActor(nn.Module):
    def __init__(self, model, config):
        super().__init__()
        self.model = model
        self.config = config
    
    def get_mu_from_conds(self, conds):
        z_mean = self.model.decoder.W_mean(conds)
        return z_mean

    def get_sigma_from_conds(self, conds):
        z_log_var = -torch.abs(self.model.decoder.W_log_var(conds))
        z_var = torch.exp(z_log_var / 2)
        return z_var
    
    def get_z_from_conds(self, conds):
        batch_size = conds.shape[0]
        z_mean = self.model.decoder.W_mean(conds)
        z_log_var = -torch.abs(self.model.decoder.W_log_var(conds))
        epsilon = torch.randn_like(z_mean)
        z_var = torch.exp(z_log_var / 2)
        z_vecs = z_mean + z_var * epsilon
        return z_vecs
    
    def forward(self, x):
        if self.config['action_type'] == 'mu':
            return self.get_mu_from_conds(x)
        elif self.config['action_type'] == 'sigma':
            return self.get_mu_from_conds(x), self.get_sigma_from_conds(x)
        elif self.config['action_type'] == 'z' or self.config['action_type'] == 'psvae':
            return self.get_z_from_conds(x)
        else:
            raise ValueError("Invalid action_type. Expected 'mu', 'sigma', or 'z'.")


if __name__ == '__main__':
    device = "cpu"
    # load vae
    ckpt = os.path.join(base_path, "PS-VAE/ckpts/zinc250k/constraint_prop_opt/epoch5.ckpt")
    vae_model = load_model(ckpt, -1)
    vae_actor = PSVAEActor(vae_model)
    input_ = torch.randn(1, 400)
    output_ = vae_actor(input_, action_type='z')
    if isinstance(output_, tuple):
        print(f"output shape: {output_[0].shape}")
        print(f"output shape: {output_[1].shape}")
    else:
        print(f"output shape: {output_.shape}")