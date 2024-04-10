import os
import sys
import torch
import json
import yaml
import argparse

import copy

import random

import numpy as np
from tqdm import tqdm

from torch_geometric.data import Data, Batch, DataLoader

sys.path.append('PS-VAE/src')
from data.bpe_dataset import BPEMolDataset, get_dataloader
from generate import load_model, parallel, gen, beam_gen

from utils.chem_utils import molecule2smiles, smiles2molecule
from utils.logger import print_log

from importance import Importance, GNNPredictor, mol_distance, Distance


random.seed(42)

def get_argparser():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--dataset_name', type=str)
    argparser.add_argument('--config', type=str, default='configs/summary.yaml')
    argparser.add_argument('--result_file', type=str, default="./data/aids/results/results_aids_1108.json.json")
    args = argparser.parse_args()
    return args

def get_covering_mapping(configs, dataset_mols, candidates):
    print("Generate covering mappings...")
    threshold = configs['threshold']

    covering_mappings = []
    dist_model = Distance(dataset_mols, threshold=threshold, dist_func=mol_distance, norm=False)
    for i in tqdm(range(len(candidates))):
        coverage_ratio, dists = dist_model.coverage(candidates[i])
        # print(molecule2smiles(candidates[i]),coverage_ratio)
        covered_index = [i for i, d in enumerate(dists) if d <= threshold]
        covering_mappings.append(covered_index)
    return covering_mappings

def greedy_counterfactual_summary_from_covering_sets(candidates, covering_mappings, k, total_num):
    """
    :param counterfactual_covering: Counterfactual -> Original graphs covered.
    :param graphs_covered_by: Original graphs -> counterfactuals that cover it.
    :param k: Number of counterfactuals in the summary.

    :return: List of indices of selected counterfactuals as summary, and the set of indices of the covered graphs.
    """

    # Greedily add the counterfactuals with maximum coverage in the remaining graphs.

    cur_index_set = [0]
    covered_graphs = covering_mappings[0]
    trace_index_set = [copy.deepcopy(cur_index_set)]
    trace_covered_graphs = [copy.deepcopy(covered_graphs)]
    print(cur_index_set, len(covered_graphs))
    for i in range(1, k):
        print("Coverage with {} counterfactuals: {}".format(i, len(covered_graphs) / total_num))

        max_coverage = 0
        max_index = -1
        max_new_cover = []
        for j in range(len(candidates)):
            if j not in cur_index_set:
                new_cover = [x for x in covering_mappings[j] if x not in covered_graphs]
                if len(new_cover) > max_coverage:
                    max_coverage = len(new_cover)
                    max_index = j
                    max_new_cover = new_cover
        if max_index != -1:
            cur_index_set.append(max_index)
            covered_graphs.extend(max_new_cover)

        trace_index_set.append(copy.deepcopy(cur_index_set))
        trace_covered_graphs.append(copy.deepcopy(covered_graphs))

    return trace_index_set, trace_covered_graphs

def minimum_distance_cost_summary(candidates, dataset_mols):
    print("Generate cost ")

    dist_matrix = []
    dist_model = Distance(dataset_mols, threshold=0, dist_func=mol_distance, norm=False)
    for i in range(len(candidates)):
        _, dists = dist_model.coverage(candidates[i])
        dist_matrix.append(dists)

    dist_matrix = np.array(dist_matrix)
    min_dists = np.min(dist_matrix, axis=0)
    return min_dists

    candidates = []
    for item in results:
        path = list(item.values())[0]
        status, score, cand = path[-1]
        candidates.append((cand, status['prediction'], status['coverage']))

    sorted_candidates = sorted(candidates, key=lambda x: x[2], reverse=True)
    sorted_candidates_smi = [x[0] for x in sorted_candidates]

    sorted_candidates_mols = parallel(smiles2molecule, sorted_candidates_smi, cpus)
    
    total_num = len(negative_mols)
    covering_mappings = get_covering_mapping(configs, negative_mols, sorted_candidates_mols)
    index_set, covered_graphs = greedy_counterfactual_summary_from_covering_sets(sorted_candidates_mols, covering_mappings, k, total_num)

    selected_candidate = [sorted_candidates_mols[i] for i in index_set]
    min_dist = minimum_distance_cost_summary(selected_candidate, negative_mols)

    return index_set, covered_graphs, min_dist
    

def get_candidate(results, max_cand_per_path=-1, max_cand_num=-1, cf=True, only_last=True):
    candidates = []
    for item in results:
        path = list(item.values())[0]
        cand_per_input = []
        if only_last:
            status, score, cand = path[-1]
            cand_per_input.append((cand, status['prediction'], status['coverage']))
        else:
            for p in path:
                status, score, cand = p
                cand_per_input.append((cand, status['prediction'], status['coverage']))
            if max_cand_per_path != -1:
                cand_per_input = sorted(cand_per_input, key=lambda x: x[2], reverse=True)
                cand_per_input = cand_per_input[:max_cand_per_path]
        candidates.extend(cand_per_input)
        candidates = list(set(candidates))

    if cf:
        cf_candidates = [x for x in candidates if x[1] >= 0.5]
        print(f"Candidate: {len(candidates)}, CF Candidate: {len(cf_candidates)} ")
        sorted_candidates = sorted(cf_candidates, key=lambda x: x[2], reverse=True)
    else:
        sorted_candidates = sorted(candidates, key=lambda x: x[2], reverse=True)

    if max_cand_num != -1:
        sorted_candidates = sorted_candidates[:max_cand_num]
    print(f'Keep {len(sorted_candidates)} candidates.')
    return sorted_candidates

def generate_summary(results, cpus, configs, negative_mols, k):
    candidates = []
    print(results)
    for item in results:
        path = list(item.values())[0]
        status, score, cand = path[-1]
        if isinstance(status, list):
            status = status[-1]
        candidates.append((cand, status['prediction'], status['coverage']))

    sorted_candidates = get_candidate(results, max_cand_per_path=configs['max_cand_per_path'], max_cand_num=configs['max_cand_num'], cf=configs['check_cf'], only_last=configs['only_last'])
    if len(sorted_candidates) == 0:
        return {}, [], [], np.array(0), np.array(1)
    sorted_candidates_smi = [x[0] for x in sorted_candidates]

    sorted_candidates_mols = parallel(smiles2molecule, sorted_candidates_smi, cpus)
    
    max_k = max(k) if isinstance(k, list) else k
    total_num = len(negative_mols)
    covering_mappings = get_covering_mapping(configs, negative_mols, sorted_candidates_mols)
    index_set, covered_graphs = greedy_counterfactual_summary_from_covering_sets(sorted_candidates_mols, covering_mappings, max_k, total_num)

    results = {}
    coverage_list = []
    min_dist_list = []
    for t in k:
        k_index_set = index_set[t-1]
        k_covered_graphs = covered_graphs[t-1]

        selected_candidate = [sorted_candidates_mols[i] for i in k_index_set]
        min_dist = minimum_distance_cost_summary(selected_candidate, negative_mols)

        coverage = len(k_covered_graphs) / total_num
        coverage_list.append(coverage)
        min_dist_list.append(min_dist.mean())

        result = {
            'Top': t,
            'selected_index': k_index_set,
            'Coverage': coverage,
            'covered_graph_number': len(k_covered_graphs),
            'total_num': total_num,
            'Cost Size': min_dist.shape,
            'Cost': {
                'mean': min_dist.mean(),
                'max': min_dist.max(),
                'min': min_dist.min()
            }
        }
        results[t] = result
        print(f"Top {t}: {result} /////////////////")
    
    mean_coverage = coverage_list[2]    # np.mean(coverage_list)
    mean_min_dist = min_dist_list[2]    # np.mean(min_dist_list)

    return results, index_set, covered_graphs, mean_coverage, mean_min_dist


# python summary.py --dataset_name aids --config "./configs/config.yaml" --result_file "./data/aids/results/results_aids_1108.json"
        
if __name__ == '__main__':

    args = get_argparser()

    configs = yaml.load(open(args.config), Loader=yaml.FullLoader)
    configs['dataset_name'] = args.dataset_name
    print(configs)

    cpus = configs['cpu']
    gpus = configs['gpu']
    k = configs['k']
    device = torch.device('cuda:{}'.format(gpus))

    # load vae
    ckpt= configs['vae_model']
    vae_model = load_model(ckpt, gpus)
    vae_model.eval()

    # load gnn
    dataset = configs['dataset_name']
    dataset_name = configs['dataset_name'].split('_')[0]
    mapping_info = json.load(open('data/{}_mapping_info.json'.format(dataset_name)))
    node_mapping = mapping_info['keep_node_mapping']
    gnn_ckpt = configs['gnn_model'].format(dataset_name=dataset_name)
    predictor = GNNPredictor(gnn_ckpt, dataset_name, node_mapping, device)

    if os.path.exists(f'data/{dataset}.pt'):
        preprocess_data = torch.load(f'data/{dataset}.pt')
        original_smis = preprocess_data['original_smis']
        predictions = preprocess_data['predictions']
        original_mols = preprocess_data['original_mols']
        negative_idx = preprocess_data['negative_idx']
        negative_mols = preprocess_data['negative_mols']
    else:
        print("There is no preprocessed data!")
        exit(0)

    result_file = args.result_file
    if os.path.exists(result_file):
        results = [json.loads(x) for x in open(result_file)]
    else:
        print("No results file found!")
        sys.exit(0)

    sorted_candidates = get_candidate(results, max_cand_per_path=configs['max_cand_per_path'], max_cand_num=configs['max_cand_num'], cf=configs['check_cf'], only_last=configs['only_last'])
    if len(sorted_candidates) == 0:
        print("No candidates!")
        sys.exit(0)
    sorted_candidates_smi = [x[0] for x in sorted_candidates]

    sorted_candidates_mols = parallel(smiles2molecule, sorted_candidates_smi, cpus)
    
    total_num = len(negative_mols)
    covering_mappings = get_covering_mapping(configs, negative_mols, sorted_candidates_mols)

    print(f"len mols: {len(sorted_candidates_mols)}")
    print(f"len neg mols: {len(negative_mols)}")

    max_k = max(k)
    trace_index_set, trace_covered_graphs = greedy_counterfactual_summary_from_covering_sets(sorted_candidates_mols, covering_mappings, max_k, total_num)
        
    summary = {}
    coverage_list = []
    min_dist_list = []
    for t in k:
        print('='*20 + f"Top {t}: " + '='*20)
        k_index_set = trace_index_set[t-1]
        k_covered_graphs = trace_covered_graphs[t-1]
        print('selected_index', k_index_set)
        print('Coverage: ',len(k_covered_graphs) / total_num, '| covered_graph_number: ', len(k_covered_graphs), '| total_num: ', total_num)

        selected_candidate = [sorted_candidates_mols[i] for i in k_index_set]
        min_dist = minimum_distance_cost_summary(selected_candidate, negative_mols)
        print('Cost Size: ', min_dist.shape)
        print('Cost: ', min_dist.mean(), min_dist.max(), min_dist.min())
        selected_candidate_smiles = [molecule2smiles(x) for x in selected_candidate]

        coverage = len(k_covered_graphs) / total_num
        coverage_list.append(coverage)
        min_dist_list.append(min_dist.mean())

        result = {
            'Top': t,
            'selected_index': k_index_set,
            'Coverage': coverage,
            'covered_graph_number': len(k_covered_graphs),
            'total_num': total_num,
            'Cost Size': min_dist.shape,
            'Cost': {
                'mean': min_dist.mean(),
                'max': min_dist.max(),
                'min': min_dist.min()
            },
            'candidates': selected_candidate_smiles,
            'cost': min_dist.tolist()
        }
        summary[t] = result

    save_file = args.result_file.replace('.json', '_summary.json')
    json.dump(summary, open(save_file, 'w'), indent=4)
    
        