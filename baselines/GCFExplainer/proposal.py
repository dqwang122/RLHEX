import os
import dgl
import math
import torch
import rdkit
import random
import pickle
from rdkit import Chem
from abc import ABC, abstractmethod
from torch.utils.data import DataLoader
import torch.nn.functional as F

from torch_geometric.data import Batch
from util import MolFromGraphs

from datasets.utils import load_vocab
from datasets.datasets import ImitationDataset
from common.utils import sample_idx
from common.chem import mol_to_geo, check_validity, \
                          Skeleton, break_bond, combine


class Proposal(ABC):
    def __init__(self, data_dir, vocab_name, vocab_size=1000, max_molecule_size=200, max_sampling_time=20, node_mapping=None):
        self.dataset = None # proposal records
        self.max_molecule_size = max_molecule_size
        self.max_sampling_time = max_sampling_time
        self.vocab = load_vocab(data_dir, vocab_name, vocab_size)
        self.node_mapping = node_mapping
        if node_mapping:
            self.atom_type = {x:i for i,x in enumerate(node_mapping)}

    @abstractmethod
    def get_pred(self, graphs):
        '''
        get prediction of editing actions
        @params:
            graphs (list): molecular graphs, torch_geometric data
        @return:
            pred_act (torch.FloatTensor): (batch_size, 2)
            pred_del (torch.FloatTensor): (tot_n_edge, 2)
            pred_add (torch.FloatTensor): (tot_n_node, 2)
            pred_arm (torch.FloatTensor): (tot_n_node, vocab_size)
        '''
        raise NotImplementedError

    def get_prob(self, graphs):
        '''
        get probability of editing actions
        @params:
            graphs (list): molecular graphs, DGLGraphs
        @return:
            prob_act (list): (batch_size, 2)
            prob_del (list): (batch_size, n_edge)
            prob_add (list): (batch_size, n_node)
            prob_arm (list): (batch_size, n_node, vocab_size)
        '''
        pred_act, pred_del, pred_add, \
            pred_arm = self.get_pred(graphs)
        pred_act = F.softmax(pred_act, dim=1) # (batch_size, 2)
        pred_del = F.softmax(pred_del, dim=1) # (tot_n_edge, 2)
        pred_add = F.softmax(pred_add, dim=1) # (tot_n_node, 2)
        pred_arm = F.softmax(pred_arm, dim=1) # (tot_n_node, vocab_size)

        prob_act = pred_act.tolist()
        prob_del, prob_add, prob_arm = [], [], []
        off_edge, off_node = 0, 0
        for g in graphs:
            n_edge = g.num_edges
            n_node = g.num_nodes
            p_del = pred_del[off_edge:off_edge+n_edge][:,1] # (n_edge,)
            p_add = pred_add[off_node:off_node+n_node][:,1] # (n_node,)
            p_arm = pred_arm[off_node:off_node+n_node].tolist() # (n_node, vocab_size)
            p_del = (p_del / (p_del.sum() + 1e-6)).tolist()
            p_add = (p_add / (p_add.sum() + 1e-6)).tolist()
            prob_del.append(p_del)
            prob_add.append(p_add)
            prob_arm.append(p_arm)
            off_edge += n_edge
            off_node += n_node
        return prob_act, prob_del, prob_add, prob_arm

    def propose(self, mols, backward=False):
        '''
        @params:
            mols : molecules to edit
        @return:
            new_mols     : proposed new molecules
            fixings      : fixing propotions for each proposal
        '''
        ### forward proposal: g(x|x')
        fixings = [1. for _ in mols]
        graphs = [mol_to_geo(mol, self.atom_type) for mol in mols]
        prob_act, prob_del, prob_add, \
            prob_arm = self.get_prob(graphs)
        
        new_mols, graphs_ = [], [] # var_: for computing fixings
        actions,  del_idxs,  add_idxs,  arm_idxs  = [], [], [], []
        actions_, del_idxs_, add_idxs_, arm_idxs_ = [], [], [], []
        for i, mol in enumerate(mols):
            not_change = True
            cnt = 0
            while not_change and cnt < self.max_sampling_time:
                action = sample_idx(prob_act[i])
                del_idx = sample_idx(prob_del[i])
                add_idx = sample_idx(prob_add[i])
                arm_idx = sample_idx(prob_arm[i][add_idx])

                not_change = False
                if action == 0: # del
                    u = graphs[i].edge_index[0][del_idx].item()
                    v = graphs[i].edge_index[1][del_idx].item()
                    try: 
                        skeleton, old_arm = break_bond(mol, u, v)
                        if skeleton.mol.GetNumBonds() <= 0: 
                            raise ValueError
                        new_mol = skeleton.mol
                    except ValueError:
                        new_mol = None
                    
                    if check_validity(new_mol):
                        if backward:
                            old_smiles = Chem.MolToSmiles(
                                old_arm.mol, rootedAtAtom=old_arm.v)
                            new_g = mol_to_geo(new_mol, self.atom_type)
                            action_ = 1 # backward: add
                            del_idx_ = None
                            add_idx_ = skeleton.u
                            arm_idx_ = self.vocab.smiles2idx.get(old_smiles)
                            fixings[i] *= prob_act[i][action]
                            fixings[i] *= prob_del[i][del_idx]
                    else:
                        not_change = True

                elif action == 1: # add
                    new_arm = self.vocab.arms[arm_idx]
                    skeleton = Skeleton(mol, u=add_idx,
                        bond_type=new_arm.bond_type)
                    new_mol = combine(skeleton, new_arm)

                    if check_validity(new_mol) and \
                        new_mol.GetNumAtoms() <= self.max_molecule_size: # limit size
                        if backward:
                            new_g = mol_to_geo(new_mol, self.atom_type)
                            u = skeleton.u
                            v = skeleton.mol.GetNumAtoms() + new_arm.v
                            u = new_g.all_edges()[0] == u
                            v = new_g.all_edges()[1] == v
                            action_ = 0 # backward: del
                            del_idx_ = (u * v).long().argmax().item()
                            add_idx_ = None
                            arm_idx_ = None
                            fixings[i] *= prob_act[i][action]
                            fixings[i] *= prob_add[i][add_idx]
                            fixings[i] *= prob_arm[i][add_idx][arm_idx]
                    else: 
                        not_change = True
                else: raise NotImplementedError

                if not_change:
                    new_mol = None
                    if backward:
                        new_g = graphs[i] # placeholder
                        action_ = None
                        del_idx_ = None
                        add_idx_ = None
                        arm_idx_ = None
                        fixings[i] = 0.

                if backward:
                    graphs_.append(new_g)
                    actions_.append(action_)
                    del_idxs_.append(del_idx_)
                    add_idxs_.append(add_idx_)
                    arm_idxs_.append(arm_idx_)

                cnt += 1


            print(cnt, not_change)
            actions.append(action)
            del_idxs.append(del_idx)
            add_idxs.append(add_idx)
            arm_idxs.append(arm_idx)
            new_mols.append(new_mol)

        ### backward proposal: g(x'|x)
        if backward:
            prob_act_, prob_del_, prob_add_, \
                prob_arm_ = self.get_prob(graphs_)
        
        for i, new_mol in enumerate(new_mols):
            if new_mol is None:
                new_mols[i] = mols[i]
                continue
            if backward:
                action_ = actions_[i]
                del_idx_ = del_idxs_[i]
                add_idx_ = add_idxs_[i]
                arm_idx_ = arm_idxs_[i]
                if action_ == 0: # del
                    fixings[i] *= prob_act_[i][action_]
                    fixings[i] *= prob_del_[i][del_idx_]
                elif action_ == 1: # add
                    fixings[i] *= prob_act_[i][action_]
                    fixings[i] *= prob_add_[i][add_idx_]
                    if arm_idx_ is None: fixings[i] *= 0.
                    else: fixings[i] *= prob_arm_[i][add_idx_][arm_idx_]
                else: raise NotImplementedError

        edits = {
            'act': actions,
            'del': del_idxs,
            'add': add_idxs,
            'arm': arm_idxs
        }
        # print(edits)
        self.dataset = ImitationDataset(graphs, edits)
        return new_mols, edits
    
    def convert_to_geo(self, mol):
        if isinstance(mol, list):
            return [mol_to_geo(m, self.atom_type) for m in mol]
        else:
            return mol_to_geo(mol, self.atom_type)

    


class Proposal_Random(Proposal):
    def __init__(self, data_dir, vocab_name, vocab_size=1000, max_molecule_size=200, max_sampling_time=20, node_mapping=None):
        super().__init__(data_dir, vocab_name, vocab_size=vocab_size, max_molecule_size=max_molecule_size, max_sampling_time=max_sampling_time, node_mapping=node_mapping)

    def get_pred(self, graphs):
        g = Batch.from_data_list(graphs)
        pred_act = torch.zeros(len(graphs), 2)
        pred_del = torch.zeros(g.num_edges, 2)
        pred_add = torch.zeros(g.num_nodes, 2)
        pred_arm = torch.zeros(g.num_nodes, len(self.vocab.arms))
        return pred_act, pred_del, pred_add, pred_arm
    
    def enumerate_from_graph(self, graph):
        '''
        @params:
            mols : molecules to edit
        @return:
            new_mols     : proposed new molecules
            fixings      : fixing propotions for each proposal
        '''
        mol = MolFromGraphs(graph, self.node_mapping)
        new_mols = []
        actions = []
        
        for action in range(2):
            if action == 0: # delete
                for del_idx in range(graph.num_edges):
                    u = graph.edge_index[0][del_idx].item()
                    v = graph.edge_index[1][del_idx].item()

                    try: 
                        skeleton, old_arm = break_bond(mol, u, v)
                        if skeleton.mol.GetNumBonds() <= 0: 
                            raise ValueError
                        new_mol = skeleton.mol
                    except ValueError:
                        new_mol = None
                    
                    if check_validity(new_mol):
                        new_mols.append(new_mol)
                        actions.append((action, del_idx, None, None))
                        
            elif action == 1:   # add
                for add_idx in range(graph.num_edges):
                    for arm_idx in range(len(self.vocab.arms)):
                        new_arm = self.vocab.arms[arm_idx]
                        if mol.GetNumAtoms() + new_arm.mol.GetNumAtoms() > self.max_molecule_size + 2:
                            new_mol = None
                        else:
                            try:
                                skeleton = Skeleton(mol, u=add_idx, bond_type=new_arm.bond_type)
                                new_mol = combine(skeleton, new_arm)
                            except:
                                new_mol = None

                        if new_mol is not None and new_mol.GetNumAtoms() <= self.max_molecule_size and check_validity(new_mol): # limit size
                            new_mols.append(new_mol)
                            actions.append((action, None, add_idx, arm_idx))
            else:
                raise NotImplementedError

        return new_mols, actions
    
    def action_on_graph(self, graph, action):
        mol = MolFromGraphs(graph, self.node_mapping)
        action_idx = action[0]
        if action_idx == 0: # delete
            del_idx = action[1]
            u = graph.edge_index[0][del_idx].item()
            v = graph.edge_index[1][del_idx].item()

            try: 
                skeleton, old_arm = break_bond(mol, u, v)
                if skeleton.mol.GetNumBonds() <= 0: 
                    raise ValueError
                new_mol = skeleton.mol
            except ValueError:
                new_mol = None
                    
        elif action_idx == 1:   # add
            add_idx, arm_idx = action[2], action[3]
            new_arm = self.vocab.arms[arm_idx]
            
            try:
                skeleton = Skeleton(mol, u=add_idx, bond_type=new_arm.bond_type)
                new_mol = combine(skeleton, new_arm)
            except:
                new_mol = None

        elif action_idx is None:
            return graph
        
        else:
            raise NotImplementedError
        
        if new_mol is not None:
            return self.convert_to_geo(new_mol)
        else: 
            return None



