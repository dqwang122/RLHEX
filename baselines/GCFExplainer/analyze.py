import os,sys
import json

import numpy as np
from rdkit import Chem
from rdkit.Chem import Draw

import math
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import torch
import networkx as nx
from torch_geometric.utils.convert import to_networkx

from util import MolFromGraphs, valid_checking
from data import load_dataset
from gnn import load_trained_prediction



class Analyzer():
    def __init__(self, dataset_name, counterfactual_name=None, factual_name=None):
    
        graphs = load_dataset(dataset_name)
        
        mapping_info = json.load(open('data/{}/raw/mapping_info.json'.format(dataset_name)))
        self.node_mapping = mapping_info['keep_node_mapping']

        preds = load_trained_prediction(dataset_name, device='cpu')
        preds = preds.cpu().numpy()

        if counterfactual_name:
            input_graph_indices = np.array(range(len(preds)))[preds == 0]
            input_graphs = graphs[input_graph_indices.tolist()]
            self.cf_valid_cand_mol = self.load_explainations(counterfactual_name, input_graphs)
        else:
            self.cf_valid_cand_mol = None

        if factual_name:
            input_graph_indices = np.array(range(len(preds)))[preds == 1]
            input_graphs = graphs[input_graph_indices.tolist()]
            self.f_valid_cand_mol = self.load_explainations(factual_name, input_graphs)
        else:
            self.f_valid_cand_mol = None


    def print_cand(self, cand_idx, cf=False, save=False):
        if cf:
            valid_cand_mol = self.cf_valid_cand_mol
        else:
            valid_cand_mol = self.f_valid_cand_mol

        cand = valid_cand_mol[cand_idx]['cand_mol']
        cover_list = valid_cand_mol[cand_idx]['cover']
        cover_num = len(cover_list)
        print(cover_num)

        ncol = cover_num if cover_num < 4 else 4
        nrow = math.ceil(cover_num / ncol) + 1
        fig = plt.figure()
        gs = gridspec.GridSpec(nrow, ncol)
        fig_title = f"Candidate: {cand_idx}"

        empty_cols = (ncol - (cover_num % ncol)) % ncol
        left_empty_cols = empty_cols // 2

        for i in range(nrow):
            for j in range(ncol):
                if i == 0:
                    if j == 0:
                        ax = plt.subplot(gs[0, :])
                        show_mol(cand, ax=ax)
                        ax.axis("off")
                        break
                else:
                    idx = (i-1) * ncol + j - left_empty_cols
                    if idx >= cover_num:
                        ax = plt.subplot(gs[i, j])
                        ax.axis("off")
                    else:
                        ax = plt.subplot(gs[i, j])
                        cover_mol = MolFromGraphs(cover_list[idx], self.node_mapping)
                        show_mol(cover_mol, ax=ax)
                        ax.axis("off")

        
        if save:
            if cf: 
                save_name = f'cf_{cand_idx}.jpg'
            else: 
                save_name = f'f_{cand_idx}.jpg'
            fig.savefig(save_name, dpi=200)



    def load_explainations(self, summary_name, input_graphs):
        summary = torch.load(summary_name)

        valid_cand_mol = []
        for cand in summary['counterfactual_candidates']:
            graph_hasp = cand['graph_hash']
            graph = summary['graph_map'][graph_hasp]
            if valid_checking(graph, self.node_mapping):
                mol = MolFromGraphs(graph, self.node_mapping)
                covers = [input_graphs[idx] for idx in cand['input_graphs_covering_list'].coalesce().indices()[0]]
                ins = {
                    'cand_mol': mol,
                    'cand_graph': graph,
                    'cover': covers,
                }
                valid_cand_mol.append(ins)
            # mol = MolFromGraphs(graph, self.node_mapping)
            # covers = [input_graphs[idx] for idx in cand['input_graphs_covering_list'].coalesce().indices()[0]]
            # ins = {
            #     'cand_mol': mol,
            #     'cand_graph': graph,
            #     'cover': covers,
            # }
            # valid_cand_mol.append(ins)
        return valid_cand_mol
    
    def calculate_distance(self, top=10):
        dists = {}
        for i in range(min(len(self.f_valid_cand_mol), top)):
            for j in range(min(len(self.cf_valid_cand_mol), top)):
                if not self.f_valid_cand_mol[i]['cand_graph'].edge_attr is None and not self.cf_valid_cand_mol[i]['cand_graph'].edge_attr is None:
                    g1 = to_networkx(self.f_valid_cand_mol[i]['cand_graph'], node_attrs=['x'], edge_attrs=['edge_attr'], graph_attrs=['y'])
                    g2 = to_networkx(self.cf_valid_cand_mol[j]['cand_graph'], node_attrs=['x'], edge_attrs=['edge_attr'], graph_attrs=['y'])
                else:
                    g1 = to_networkx(self.f_valid_cand_mol[i]['cand_graph'], node_attrs=['x'], graph_attrs=['y'])
                    g2 = to_networkx(self.cf_valid_cand_mol[j]['cand_graph'], node_attrs=['x'], graph_attrs=['y'])
                d = nx.graph_edit_distance(g1, g2, timeout=1)
                dists[f'f_{i}_cf_{j}'] = d

        return dists
    
    # def calculate_distance2(self, top=10):
    #     dists = {}
    #     for i in range(min(len(self.f_valid_cand_mol), top)):
    #         for j in range(min(len(self.cf_valid_cand_mol), top)):
    #             g1 = to_networkx(self.f_valid_cand_mol[0]['cand_graph'], node_attrs=['x'], edge_attrs=['edge_attr'], graph_attrs=['y'])
    #             g2 = to_networkx(self.cf_valid_cand_mol[1]['cand_graph'], node_attrs=['x'], edge_attrs=['edge_attr'], graph_attrs=['y'])
    #             d = nx.graph_edit_distance(g1,g2, timeout=1, upper_bound=40)
    #             dists[f'f_{i}_cf_{i}'] = d

    #     return dists

def show_mol(mol, ax=None, size=(500,500)):
    im = Draw.MolToImage(mol, size=size)
    if ax is None:
        plt.axis("off")
        plt.imshow(im)
    else:
        ax.axis("off")
        ax.imshow(im)