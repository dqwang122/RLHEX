"""
    The file contains the PPO class to train with.
    NOTE: All "ALG STEP"s are following the numbers from the original PPO pseudocode.
            It can be found here: https://spinningup.openai.com/en/latest/_images/math/e62a8971472597f4b014c2da064f636ffe365ba3.svg
"""

from copy import deepcopy
import numpy as np
import os
import time
import yaml
from collections import defaultdict
import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from torch.distributions import MultivariateNormal
import wandb

# RLHEX: Generating Global Counterfactual Explanations for Molecular Properties using Reinforcement Learning via Human-guided Explanations


from eval_policy import eval_policy
from psvae_RL import PSVAEActor
from network import LinearWarmupCosineDecayScheduler

import multiprocessing as mp
from multiprocessing import Pool
from queue import Empty

from rdkit import Chem
import collections

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
torch.autograd.set_detect_anomaly(True)

def get_mu_from_conds(model, conds):
    z_mean = model.decoder.W_mean(conds)
    return z_mean

def test(env, actor_model):
    """
        Tests the model.

        Parameters:
            env - the environment to test the policy on
            actor_model - the actor model to load in

        Return:
            None
    """
    print(f"Testing {actor_model}", flush=True)

    # Extract out dimensions of observation and action spaces
    obs_dim = env.obs_dim
    act_dim = env.act_dim

    # Load in the actor model saved by the PPO algorithm
    if isinstance(actor_model, str):
        policy.load_state_dict(torch.load(actor_model, map_location=torch.device('cpu')))
    else:
        print(f"actor_model is not a string, it is {type(actor_model)}")
        policy = actor_model

    # Evaluate our policy with a separate module, eval_policy, to demonstrate
    # that once we are done training the model/policy with ppo.py, we no longer need
    # ppo.py since it only contains the training algorithm. The model/policy itself exists
    # independently as a binary file that can be loaded in with torch.
    return eval_policy(policy=policy, env=env)

def create_env(env):
    return env.duplicate()

class PPO:
    """
        This is the PPO class we will use as our model in main.py
    """
    def __init__(self, policy_class, env, args, **hyperparameters):
        """
            Initializes the PPO model, including hyperparameters.

            Parameters:
                policy_class - the policy class to use for our actor/critic networks.
                env - the environment to train on.
                hyperparameters - all extra arguments passed into PPO that should be hyperparameters.

            Returns:
                None
        """
        
        self.args = args
        self.config_args = None if args.config is None else yaml.load(open(args.config), Loader=yaml.FullLoader)

        self.obs_type = args.obs_type
        self._init_hyperparameters(hyperparameters)

        # Extract environment information and set base_path
        self.env = env
        self.config_str = str(self.env.config).replace("{", "").replace("}", "").replace(":", "_").replace(", ", "_").replace("'", "")
        self.model_name = os.path.join(self.args.dataset, f"{self.args.name}_{self.config_str}")
        if self.args.ckpt_dir is not None:
            base_path = self.args.ckpt_dir
        elif self.args.actor_model is not None:
            base_path = os.path.dirname(self.args.actor_model)
        else:
            base_path = os.path.join("./models", self.model_name) if self.args.name else os.path.join("./models", self.config_str)
        if not os.path.exists(base_path):
            os.makedirs(base_path)

        self.base_path = base_path
        self.env.base_path = base_path
        self.obs_dim = env.obs_dim
        self.act_dim = env.act_dim

        self.mols = env.mols
        # Initialize actor and critic networks
        self.PSVAE = env
        # # use the output of the en coder to predict the value
        # # approximate value through mean reward rollout
        device = self.env.ps_vae.device
        # TODO Change the actor into transformer or residual encoder
        # self.actor = self.env.ps_vae
        self.critic = policy_class(obs_dim=self.obs_dim, act_dim=1).to(device) # use the output of the encoder to approximate the value function
        if self.args.actor_type == 'transformer':
            self.actor = policy_class(obs_dim=self.obs_dim, act_dim=self.act_dim).to(device)       # use the mean here to sample an action
        elif self.args.actor_type == 'psvae':
            if self.args.action_type == 'psvae':
                self.actor = PSVAEActor(self.env.ps_vae.to(device),
                                        config=self.env.config)
            else:
                self.actor = PSVAEActor(deepcopy(self.env.ps_vae).to(device),
                                        config=self.env.config)

        # Initialize optimizers for actor and critic
        if not self.args.sgd:
            self.actor_optim = Adam(self.actor.parameters(), lr=self.lr)
            self.critic_optim = Adam(self.critic.parameters(), lr=self.lr_critic)
        else:
            self.actor_optim = SGD(self.actor.parameters(), lr=self.lr, momentum=0.9)
            self.critic_optim = SGD(self.critic.parameters(), lr=self.lr_critic, momentum=0.9)
        
        # Initialize learning rate scheduler ONLY FOR ACTOR
        if self.args.lr_scheduler:
            self.lr_scheduler = LinearWarmupCosineDecayScheduler(self.actor_optim, 
                                                                 warmup_steps=10,
                                                                 total_steps=self.args.total_timesteps,
                                                                 min_lr=5e-7)

        # load params if dir exists
        if self.args.ckpt_dir is not None and self.args.rand_actor is False:
            print(f"//// Loading models from {self.args.ckpt_dir} ////")
            if self.args.mode == 'train':
                prefix = 'ppo'
            elif self.args.mode == 'test':
                if self.args.most_recent:
                    prefix = 'ppo'
                else:
                    prefix = 'best_ppo' # "best_mean_cov_ppo"
            print(f"prefix: {prefix}")
            self.actor.load_state_dict(torch.load(os.path.join(self.args.ckpt_dir, f'{prefix}_actor.pt'), 
                                                  map_location=torch.device('cpu')))
            self.critic.load_state_dict(torch.load(os.path.join(self.args.ckpt_dir, f'{prefix}_critic.pt'), 
                                                   map_location=torch.device('cpu')))
            print(f"CKPT NOT NONE!!!!!!")
        elif self.args.rand_actor:
            print(f"//// Random actor for ablation study ////")
        else:
            print(f"CKPT_DIR IS NONE")

        # Initialize the covariance matrix used to query the actor for actions
        # our covariance matrix are the sigmas corresponding to each mu of the PS-VAE
        # TODO: Confirm that torch.diag() is the correct function to use here

        # self.cov_mat = self.PSVAE.get_sigma # This logger will help us with printing out summaries of each iteration
        self.cov_var = torch.full(size=(self.act_dim,), fill_value=0.5)
        self.cov_mat = torch.diag(self.cov_var).to(device)
        
        self.best_coverage = float('-inf')
        self.best_mean_coverage = float('-inf')
        self.logger = {
            'delta_t': time.time_ns(),
            't_so_far': 0,          # timesteps so far
            'i_so_far': 0,          # iterations so far
            'batch_lens': [],       # episodic lengths in batch
            'batch_rews': [],       # episodic returns in batch
            'actor_losses': [],     # losses of actor network in current iteration
            'critic_loss': [],      # loss of critic network in current iteration
        }
        if args.wandb:
            wandb.init(project="ppo-psvae", group=f"{self.args.dataset}_2nd", name=args.name, 
                       resume="allow", id=args.run_id)
            wandb.config.update(hyperparameters)
        
        self.mol_rewards = self.init_mol_rews(self.mols['train']['negative'])
        self.env.mol_rewards = self.mol_rewards
    
    def init_mol_rews(self, mols):
        self.mol_rewards = {}
        for idx, mol in enumerate(mols):
            mol_smi = Chem.MolToSmiles(mol)
            self.mol_rewards[mol_smi] = {'count': 0, 'total_reward': 0, 'idx': idx}

        # print(f"MOL REWARDS: {self.mol_rewards}")
        return self.mol_rewards

    def update_mol_rews(self, mol, reward):
        print(f"Updating mol rewards for {mol} with reward {reward}")
        self.mol_rewards[mol]['total_reward'] += reward
        self.mol_rewards[mol]['count'] += 1
        return self.mol_rewards
        
    def learn(self, total_timesteps):
        """
            Train the actor and critic networks. Here is where the main PPO algorithm resides.

            Parameters:
                total_timesteps - the total number of timesteps to train for

            Return:
                None
        """
        print(f"Learning... Running {self.max_timesteps_per_episode} timesteps per episode, ", end='')
        print(f"{self.timesteps_per_batch} timesteps per batch for a total of {total_timesteps} timesteps")
        t_so_far = 0 # Timesteps simulated so far
        i_so_far = 0 # Iterations ran so far
        while t_so_far < total_timesteps:                                                                       # ALG STEP 2
            # Autobots, roll out (just kidding, we're collecting our batch simulations here)
            batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens = self.rollout()               # ALG STEP 3

            # Calculate how many timesteps we collected this batch
            t_so_far += np.sum(batch_lens)

            # Increment the number of iterations
            i_so_far += 1

            # Logging timesteps so far and iterations so far
            self.logger['t_so_far'] = t_so_far
            self.logger['i_so_far'] = i_so_far

            print(f"ROLLOUT DONE, {batch_obs.shape}, {batch_acts.shape}, {batch_log_probs.shape}, {batch_rtgs.shape}, {len(batch_lens)}")

            # Calculate advantage at k-th iteration
            V, _ = self.evaluate(batch_obs, batch_acts)
            device = V.device
            batch_rtgs = batch_rtgs.to(device).detach()
            print(f"batch_rtgs: {batch_rtgs}, V: {V}")
            A_k = batch_rtgs - V.detach()                                                                       # ALG STEP 5

            # One of the only tricks I use that isn't in the pseudocode. Normalizing advantages
            # isn't theoretically necessary, but in practice it decreases the variance of 
            # our advantages and makes convergence much more stable and faster. I added this because
            # solving some environments was too unstable without it.
            A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10)

            print(f"A_k: {A_k}")

            for _ in range(self.n_updates_per_iteration):
                # Detach or clone the tensors used to compute the losses
                batch_obs_iter = batch_obs.clone().detach()
                batch_acts_iter = batch_acts.clone().detach()
                batch_rtgs_iter = batch_rtgs.clone().detach()
                A_k_iter = A_k.clone().detach()

                # Calculate V_phi and pi_theta(a_t | s_t)
                V, curr_log_probs = self.evaluate(batch_obs_iter, batch_acts_iter)

                # Calculate the ratio pi_theta(a_t | s_t) / pi_theta_k(a_t | s_t)
                ratios = torch.exp(curr_log_probs - batch_log_probs.detach())

                # Calculate surrogate losses.
                print(torch.isnan(ratios).any(), torch.isnan(A_k_iter).any())
                print(ratios.min(), ratios.max(), A_k_iter.min(), A_k_iter.max())
                surr1 = ratios * A_k_iter
                surr2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * A_k_iter

                # Calculate actor and critic losses.
                actor_loss = (-torch.min(surr1, surr2)).mean()
                if self.args.entropy_reg is not None:
                    entropy_reg = curr_log_probs.mean()
                    actor_loss -= self.args.entropy_reg * entropy_reg

                critic_loss = nn.MSELoss()(V, batch_rtgs_iter)

                # Calculate gradients and perform backward propagation for actor network
                self.actor_optim.zero_grad()
                actor_loss.backward(retain_graph=True)
                self.actor_optim.step()
                if self.args.lr_scheduler:
                    self.lr_scheduler.step()
                    print(f"Learning rate: {self.actor_optim.param_groups[0]['lr']}")
                    exit()

                # Calculate gradients and perform backward propagation for critic network
                self.critic_optim.zero_grad()
                critic_loss.backward()
                self.critic_optim.step()

                # Log actor loss
                self.logger['critic_loss'].append(critic_loss.detach())
                self.logger['actor_losses'].append(actor_loss.detach())

            # Print a summary of our training so far
            mean_coverage = np.mean( self.logger['coverages'])
            self._log_summary()

            # if debug, end here
            if self.args.debug:
                index_set, covered_graphs, coverage, min_dist = test(self.env, self.actor)
                break

            # Save our model if it's time
            # Append the config string to the filename
            actor_name = f"ppo_actor.pt"
            critic_name = f"ppo_critic.pt"
            if i_so_far % self.save_freq == 0:
            # 	return self.actor, self.critic
                torch.save(self.actor.state_dict(), os.path.join(self.base_path, actor_name))
                torch.save(self.critic.state_dict(), os.path.join(self.base_path, critic_name))
                print(f"Saved actor and critic models at iteration {i_so_far}")
            
            if i_so_far % self.args.eval_every == 0:
                results, index_set, covered_graphs, coverage, min_dist = test(self.env, self.actor)
                if coverage > self.best_coverage:
                    self.best_coverage = coverage
                    torch.save(self.actor.state_dict(), os.path.join(self.base_path, f"best_{actor_name}"))
                    torch.save(self.critic.state_dict(), os.path.join(self.base_path, f"best_{critic_name}"))
                    print(f"Saved best actor and critic models at iteration {i_so_far} and coverage {coverage}")
                if mean_coverage > self.best_mean_coverage:
                    self.best_mean_coverage = mean_coverage
                    torch.save(self.actor.state_dict(), os.path.join(self.base_path, f"best_mean_cov_{actor_name}"))
                    torch.save(self.critic.state_dict(), os.path.join(self.base_path, f"best_mean_cov_{critic_name}"))
                    print(f"Saved best mean coverage actor and critic models at iteration {i_so_far} and mean coverage {mean_coverage}")
                # save args as .yml file
                args_dict = vars(self.args)
                with open(os.path.join(self.base_path, 'config.yml'), 'w') as f:
                    yaml.dump(args_dict, f, default_flow_style=False)
                print(f"---- Coverage: {coverage}, Min Dist: {min_dist.mean()} -----")
                if self.args.wandb:
                    wandb.log({'coverage': coverage, 
                               'min_dist': min_dist.mean()})

    def rollout(self):
        """
            Too many transformers references, I'm sorry. This is where we collect the batch of data
            from simulation. Since this is an on-policy algorithm, we'll need to collect a fresh batch
            of data each time we iterate the actor/critic networks.

            Parameters:
                None

            Return:
                batch_obs - the observations collected this batch. Shape: (number of timesteps, dimension of observation)
                batch_acts - the actions collected this batch. Shape: (number of timesteps, dimension of action)
                batch_log_probs - the log probabilities of each action taken this batch. Shape: (number of timesteps)
                batch_rtgs - the Rewards-To-Go of each timestep in this batch. Shape: (number of timesteps)
                batch_lens - the lengths of each episode this batch. Shape: (number of episodes)
        """
        # Batch data. For more details, check function header.
        batch_obs = []
        batch_acts = []
        batch_log_probs = []
        batch_rews = []
        batch_rtgs = []
        batch_lens = []
        all_results = []
        mol_indexes = []

        # Episodic data. Keeps track of rewards per episode, will get cleared
        # upon each new episode
        ep_rews = []

        t = 0 # Keeps track of how many timesteps we've run so far this batch

        # Keep simulating until we've run more than or equal to specified timesteps per batch (batch_size for each rollout)
        while t < self.timesteps_per_batch:
            ep_rews = [] # rewards collected per episode

            # Reset the environment. Note that obs is short for observation. 
            self.env.reset()
            done = False

            # while True:
                # try:
            z_vec, emb, mol  = self.env.sample_mol('train', use_ucb=self.args.ucb)
            original_mol_smi = Chem.MolToSmiles(mol)
            mol_indexes.append(self.mol_rewards[original_mol_smi]['idx'])
            obs = {'mol': mol, 'emb': emb, 'z': z_vec}
                    # break  # if the above lines succeed, break the loop
                # except Exception as e:
                    # print(f"An error occurred: {e}. Retrying...")

            # Run an episode for a maximum of max_timesteps_per_episode timesteps
            for ep_t in range(self.max_timesteps_per_episode):
                # Calculate action and make a step in the env. 
                # Note that rew is short for reward.
                # TODO: Think a little bit more about mean reward per step etc

                obs_ori = obs[self.obs_type]
                print(f"obs_type: {self.obs_type}, obs_ori: {obs_ori.shape}")
                action, log_prob = self.get_action(obs[self.obs_type])
                
                try:
                    obs, rew, done, result = self.env.step(obs[self.obs_type], action, mol)
                except:
                    print(f"Error with action ---")
                    continue
                
                if len(obs_ori.shape) == 1:
                    obs_ori = obs_ori.unsqueeze(0)
                elif len(action.shape) == 1:
                    action = action.unsqueeze(0)
                elif len(log_prob.shape) == 0:
                    log_prob = log_prob.unsqueeze(0)
                else:
                    print(f"xxxxxx --- no valid log_prob --- xxxxx")
                    continue
                    
                # except:
                #     print(f"Error with action ---")
                #     print(f"obs: {obs}")
                #     continue
                
                batch_obs.append(obs[self.obs_type])
                ep_rews.append(rew)
                batch_acts.append(action)
                batch_log_probs.append(log_prob)
                all_results.extend(result)
                
                t += 1 # Increment timesteps ran this batch so far
                # print(f"{ep_t}. done: {done}, action shape: {action.shape}")
                # print("\n" + '-'*30)
                # print(f"Time Step: {t}")
                # print(f"Episode Time Step: {ep_t}")
                # print(f"Done: {'Yes' if done else 'No'}")
                # print(f"Reward: {rew}")
                # print('-'*30 + "\n")

                # If the environment tells us the episode is terminated, break
                if done:
                    print(f"{'-/-/'*5} Episode {len(batch_lens)} finished after {ep_t+1} timesteps with reward {sum(ep_rews)} {'-/-/'*5}", flush=True)
                    break
            
            # print(f"{'-/-/'*5} Episodtoe {len(batch_lens)} finished after {ep_t+1} timesteps with reward {sum(ep_rews)} {'-/-/'*5}", flush=True)

            # Track episodic lengths and rewards
            batch_lens.append(ep_t + 1)
            batch_rews.append(ep_rews)
            # check if nan in ep_rews
            if np.isnan(ep_rews).any():
                print(f"nan in ep_rews: {ep_rews}")
            # update molecule_rewards
            # print(f"---- UPDATING MOL REWARDS: {original_mol_smi}, {np.mean(ep_rews)}")
            self.env.mol_rewards = self.update_mol_rews(original_mol_smi, np.mean(ep_rews))

        # Reshape data as tensors in the shape specified in function description, before returning
        # batch_obs = torch.tensor(batch_obs, dtype=torch.float)
        batch_obs = torch.stack(batch_obs)
        batch_acts = torch.stack(batch_acts)
        batch_log_probs = torch.stack(batch_log_probs)
        # check if there are any nans in batch_rews
        # Iterate over each list in batch_rews and check for NaN values
        for i, rews in enumerate(batch_rews):
            if np.isnan(rews).any():
                print(f"nan in batch_rews list {i}: {rews}, mol: {mol_indexes[i]}")
                exit()
        batch_rtgs = self.compute_rtgs(batch_rews)
        coverages = [r[2]['coverage'] for r in all_results]
        predictions = [r[2]['prediction'] for r in all_results]
        # ALG STEP 4

        # Log the episodic returns and episodic lengths in this batch.
        self.logger['batch_rews'] = batch_rews
        self.logger['batch_lens'] = batch_lens
        self.logger['coverages'] = coverages
        self.logger['predictions'] = predictions
        self.logger['mol_indexes'] = mol_indexes

        return batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens

    def compute_rtgs(self, batch_rews):
        """
            Compute the Reward-To-Go of each timestep in a batch given the rewards.

            Parameters:
                batch_rews - the rewards in a batch, Shape: (number of episodes, number of timesteps per episode)

            Return:
                batch_rtgs - the rewards to go, Shape: (number of timesteps in batch)
        """
        # The rewards-to-go (rtg) per episode per batch to return.
        # The shape will be (num timesteps per episode)
        batch_rtgs = []

        # Iterate through each episode
        for ep_rews in reversed(batch_rews):

            discounted_reward = 0 # The discounted reward so far

            # Iterate through all rewards in the episode. We go backwards for smoother calculation of each
            # discounted return (think about why it would be harder starting from the beginning)
            for i, rew in enumerate(reversed(ep_rews)):
                discounted_reward = rew + discounted_reward * self.gamma
                print(f"rew: {rew}, discounted_reward: {discounted_reward}")
                # check for nan
                if np.isnan(discounted_reward):
                    print(f"nan in discounted_reward {discounted_reward}")
                    exit()
                batch_rtgs.insert(0, discounted_reward)

        # Convert the rewards-to-go into a tensor
        batch_rtgs = torch.tensor(batch_rtgs, dtype=torch.float)

        return batch_rtgs

    def get_action(self, obs):
        """
            Queries an action from the actor network, should be called from rollout.

            Parameters:
                obs - the observation at the current timestep

            Return:
                action - the action to take, as a numpy array
                log_prob - the log probability of the selected action in the distribution
        """
        # Query the actor network for a mean action
        # TODO: Make sure the actor makes sense
        """
        obs -> graph_embedding (shape): (400)
        action_space -> z (shape): (56 (cotninuous))
        action -> sampled z (shape): (56)

        # check the shape of the graph_embedding
        # print(obs.shape)
        """

        # Create a distribution with the mean action and std from the covariance matrix above.
        # For more information on how this distribution works, check out Andrew Ng's lecture on it:
        # https://www.youtube.com/watch?v=JjB58InuTqM
        # TODO: figure out how we want cov_mat to be calculated ✅
        # TODO: figure out where we want to intervene on the latent space
        # TODO: i.e z, graph_embedding, mu, mu and sigma.
        # try:
        mean = self.actor(obs)
        dist = MultivariateNormal(mean, self.cov_mat)
        action = dist.sample() # Sample an action from the distribution
        log_prob = dist.log_prob(action) # Calculate the log probability for that action
        # except:
        #     print(f"Error with action --- obs shape: {obs.shape}")
        #     print(f"obs: {obs}")
        #     exit()
        # print(f" /////// log_prob: {log_prob.detach().cpu().numpy().shape} //////")
        # print(log_prob.detach().cpu().numpy())
        # if self.args.debug:
        #     save_pth = "./debug/"
        #     filename = f"log_probs_trained_{self.args.ckpt_dir != None}.npy"
        #     # if file exists load it
        #     os.makedirs(save_pth, exist_ok=True)
        #     if os.path.exists(os.path.join(save_pth, filename)):
        #         print(f"Loading {filename}")
        #         log_prob_np = np.load(os.path.join(save_pth, filename))
        #         log_prob_np = np.concatenate((log_prob_np, np.expand_dims(log_prob.detach().cpu().numpy(), axis=0)), axis=0)
        #         np.save(os.path.join(save_pth, filename), log_prob_np)
        #     else:
        #         np.save(os.path.join(save_pth, filename), np.expand_dims(log_prob.detach().cpu().numpy(), axis=0))

        # Return the sampled action and the log probability of that action in our distribution
        return action.detach(), log_prob.detach()

    def evaluate(self, batch_obs, batch_acts):
        """
            Estimate the values of each observation, and the log probs of
            each action in the most recent batch with the most recent
            iteration of the actor network. Should be called from learn.

            Parameters:
                batch_obs - the observations from the most recently collected batch as a tensor.
                            Shape: (number of timesteps in batch, dimension of observation)
                batch_acts - the actions from the most recently collected batch as a tensor.
                            Shape: (number of timesteps in batch, dimension of action)

            Return:
                V - the predicted values of batch_obs
                log_probs - the log probabilities of the actions taken in batch_acts given batch_obs
        """
        # Query critic network for a value V for each batch_obs. Shape of V should be same as batch_rtgs
        batch_obs = batch_obs.squeeze()
        V = self.critic(batch_obs).squeeze()

        # Calculate the log probabilities of batch actions using most recent actor network.
        # This segment of code is similar to that in get_action()
        mean = self.actor(batch_obs)
        # variances = self.cov_mat(batch_obs)
        # covariance_matrix = torch.diag_embed(variances)
        covariance_matrix = self.cov_mat
        dist = MultivariateNormal(mean, covariance_matrix)
        log_probs = dist.log_prob(batch_acts)

        # Return the value vector V of each observation in the batch
        # and log probabilities log_probs of each action in the batch
        return V, log_probs

    def _init_hyperparameters(self, hyperparameters):
        """
            Initialize default and custom values for hyperparameters

            Parameters:
                hyperparameters - the extra arguments included when creating the PPO model, should only include
                                    hyperparameters defined below with custom values.

            Return:
                None
        """
        # Initialize default values for hyperparameters
        # Algorithm hyperparameters
        self.timesteps_per_batch = 5                 # Number of timesteps to run per batch
        self.max_timesteps_per_episode = 1600           # Max number of timesteps per episode
        self.n_updates_per_iteration = 5                # Number of times to update actor/critic per iteration
        self.lr = 0.005                                 # Learning rate of actor optimizer
        self.lr_critic = 5e-6
        self.gamma = 0.95                               # Discount factor to be applied when calculating Rewards-To-Go
        self.clip = 0.2                                 # Recommended 0.2, helps define the threshold to clip the ratio during SGA

        # Miscellaneous parameters
        self.render = True                              # If we should render during rollout
        self.render_every_i = 10                        # Only render every n iterations
        self.save_freq = 10                             # How often we save in number of iterations
        self.seed = 420                                # Sets the seed of our program, used for reproducibility of results

        # Change any default values to custom values for specified hyperparameters
        for param, val in hyperparameters.items():
            exec('self.' + param + ' = ' + str(val))

        # Sets the seed if specified
        if self.seed != None:
            # Check if our seed is valid first
            assert(type(self.seed) == int)

            # Set the seed 
            torch.manual_seed(self.seed)
            print(f"Successfully set seed to {self.seed}")

    def _log_summary(self):
        """
            Print to stdout what we've logged so far in the most recent batch.

            Parameters:
                None

            Return:
                None
        """
        # Calculate logging values. I use a few python shortcuts to calculate each value
        # without explaining since it's not too important to PPO; feel free to look it over,
        # and if you have any questions you can email me (look at bottom of README)
        delta_t = self.logger['delta_t']
        self.logger['delta_t'] = time.time_ns()
        delta_t = (self.logger['delta_t'] - delta_t) / 1e9
        delta_t = str(round(delta_t, 2))

        t_so_far = self.logger['t_so_far']
        i_so_far = self.logger['i_so_far']
        avg_ep_lens = np.mean(self.logger['batch_lens'])
        print(self.logger['batch_rews'])
        avg_ep_rews = np.mean([item for sublist in self.logger['batch_rews'] for item in sublist])
        avg_actor_loss = torch.cat([loss.unsqueeze(0) for loss in self.logger['actor_losses']]).mean().item()
        avg_critic_loss = torch.cat([loss.unsqueeze(0) for loss in self.logger['critic_loss']]).mean().item()

        # Round decimal places for more aesthetic logging messages
        avg_ep_lens = round(avg_ep_lens, 2)
        avg_ep_rews = round(avg_ep_rews, 2)
        avg_actor_loss = round(avg_actor_loss, 5)
        mean_coverage = np.mean(self.logger['coverages'])
        mean_predictions = np.mean(self.logger['predictions'])

        # Calculate the percentage of positive rewards
        flat_rewards = [item for sublist in self.logger['batch_rews'] for item in sublist]
        print(f"flat_rewards: {flat_rewards}")
        n_rewards = len(flat_rewards)
        n_rewards_pos = len([r for r in flat_rewards if r > 0])
        p_rewards_pos = n_rewards_pos/n_rewards

        # Print logging statements
        print(flush=True)
        print(f"-------------------- Iteration #{i_so_far} --------------------", flush=True)
        print(f"Average Episodic Length: {avg_ep_lens}", flush=True)
        print(f"Average Episodic Return: {avg_ep_rews}", flush=True)
        print(f"Average Loss: {avg_actor_loss}", flush=True)
        print(f"Timesteps So Far: {t_so_far}", flush=True)
        print(f"Iteration took: {delta_t} secs", flush=True)
        print(f"Mol Indexes: {self.logger['mol_indexes']}", flush=True)
        print(f"mean_predictions: {mean_predictions}, mean_coverage: {mean_coverage}")
        # print(f"Mol rews: {self.mol_rewards}", flush=True)
        print(f"n mols > 0: {len([k for k, v in self.mol_rewards.items() if v['total_reward'] > 0])}", flush=True)
        print(f"p_pos_rewards: {p_rewards_pos}", flush=True)
        # wait for 10 seconds
        # time.sleep(15)
        print(f"------------------------------------------------------", flush=True)
        print(flush=True)

        # log to wandb
        if self.args.wandb:
            wandb.log({
                       'avg_ep_lens': avg_ep_lens, 
                       'avg_ep_rews': avg_ep_rews, 
                       'avg_actor_loss': avg_actor_loss, 
                       'avg_critic_loss': avg_critic_loss,
                       't_so_far': t_so_far, 
                       'delta_t': delta_t,
                       'mean_coverage': mean_coverage,
                       'mean_predictions': mean_predictions,
                       'mol_indexes': self.logger['mol_indexes'],
                       'p_rewards_pos': p_rewards_pos,
                       })
        
        print(f"succefully logged to wandb")

        # Reset batch-specific logging data
        self.logger['batch_lens'] = []
        self.logger['batch_rews'] = []
        self.logger['actor_losses'] = []
        self.logger['critic_loss'] = []
        self.logger['coverages'] = []
        self.logger['predictions'] = []
