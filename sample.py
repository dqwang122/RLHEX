import os
import sys
import torch
import json
import yaml
import argparse
import traceback
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

from importance import Importance, GNNPredictor, mol_distance
from tools import check_aromatic


DEBUG=True
# DEBUG=False

def get_argparser():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--config', type=str, default='configs/config.yaml')

    argparser.add_argument('--dataset_name', type=str)
    argparser.add_argument('--save_file', type=str)
    argparser.add_argument('--max_iter', type=int)
    
    argparser.add_argument('--stop_early', type=bool)
    argparser.add_argument('--cpu', type=int, default=2)
    argparser.add_argument('--gpu', type=int, default=0)
    argparser.add_argument('--seed', type=int, default=42)
    
    args = argparser.parse_args()
    return args


def beam_gen_2(model, z, beam, max_atom_num, add_edge_th, temperature, constraint_mol=None):
    gens = [gen(model, z, max_atom_num, add_edge_th, temperature, constraint_mol) for _ in range(beam)]
    mols = [model.return_data_to_mol(g) for g in gens]
    return mols

def annealing_sample(vae_model, predictor, ori_mol, ori_smi, negative_mols, configs):
    if check_aromatic(ori_mol):
        return []
    mol, smi = copy.deepcopy(ori_mol), copy.deepcopy(ori_smi)
    scorer = Importance(negative_mols, predictor, threshold=configs['threshold'], norm=False)
    cur_temp = configs['annealing_temp']

    prev_score, status, _ = scorer.get_importance(mol, target_class=1, keys=configs['importance']['key'], weights=configs['importance']['weight'])

    begin = (mol, smi, prev_score, status)
    paths = [begin]
    
    best_candidate, best_score, best_status = mol, prev_score, status
    for i in range(configs['max_iter']):
        print(f"max_iter: {configs['max_iter']}")
        try:
            z = vae_model.get_z_from_mol(mol)

            # mol = vae_model.inference_single_z(z, max_atom_num, add_edge_th, temperature)
            # mols = beam_gen_2(vae_model, z, configs['beam_size'], configs['max_atom_num'], configs['add_edge_th'], cur_temp, constraint_mol=ori_mol)
            mols = beam_gen_2(vae_model, z, configs['beam_size'], configs['max_atom_num'], configs['add_edge_th'], configs['temperature'], constraint_mol=mol)
            
            scores = []
            for t, m in enumerate(mols):

                # check whether there is rdkit.Chem.rdchem.BondType.AROMATIC
                if check_aromatic(m):
                    continue

                score, status, details = scorer.get_importance(m, target_class=1, keys=configs['importance']['key'], weights=configs['importance']['weight'])
                scores.append((m, score, status))

                if DEBUG:
                    print('*'*20 + 'Sample {}'.format(t) + '*'*20)
                    print('Original Mol: ', smi)
                    print('New Mol: ', molecule2smiles(m))
                    print('Distance: ',mol_distance(ori_mol, m))
                    print('Score: ', score)
                    print('Status: ', status)

            if scores == []:
                continue

            scores = sorted(scores, key=lambda x: x[1], reverse=True)

            best_candidate, best_score, best_status = scores[0][0], scores[0][1], scores[0][2]
            accept_rate = min(np.exp((best_score - prev_score) / cur_temp), 1)
            print('Cur status: {}'.format(scores[0][2]))
            print('Iter {}, cur score: {}, prev score: {}, accept rate: {}'.format(i, best_score, prev_score, accept_rate))
            ratio = np.random.rand()
            if accept_rate > ratio:
                print('Accept!')
                mol = best_candidate
                prev_score = best_score
                smi = molecule2smiles(mol)
                paths.append((mol, smi, best_score, best_status))
            else:
                print('Reject!')

        except Exception as e:
            print(e)
            # print(repr(e))
            print(traceback.format_exc())
            print('pop last mol')
            paths.pop()
            if len(paths) != 0:
                print('recover from current last mol')
                mol, smi, prev_score, _ = paths[-1]
            else:
                print('recover from beginning')
                paths = [begin]
            continue
            
        finally:
            print("-"*40 + "Iter {} Done".format(i) + "-"*40)
            if i % 10 == 0:
                cur_temp = cur_temp / 2
            if configs['stop_early']:
                for k, v in configs['stop_condition'].items():
                    if k in best_status:
                        if best_status[k] >= v:
                            print('Early Stop!')
                            return paths
    
    return paths

        
# python sample.py --config configs/baseline.yaml --save_file baseline.json --gpu 0 --stop_early True
# python sample.py --config configs/config_dipole.yaml --save_file dipole.json --gpu 0

if __name__ == '__main__':

    args = get_argparser()

    random.seed(args.seed)

    cpus = args.cpu
    gpus = args.gpu
    device = torch.device('cuda:{}'.format(gpus))

    configs = yaml.load(open(args.config), Loader=yaml.FullLoader)
    custom_configs = vars(args)
    custom_configs = {k:v for k, v in custom_configs.items() if v is not None}
    print(custom_configs)
    configs.update(custom_configs)
    print(configs)
    print(f"max iter: {configs['max_iter']} ------ ")

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
    
    # get negative samples (p(class=0) >= 0.5)
    if os.path.exists(f'data/{dataset}.pt'):
        print('Loading Dataset...')
        preprocess_data = torch.load(f'data/{dataset}.pt')
        original_smis = preprocess_data['original_smis']
        predictions = preprocess_data['predictions']
        original_mols = preprocess_data['original_mols']
        negative_idx = preprocess_data['negative_idx']
        negative_mols = preprocess_data['negative_mols']
    else:
        print('Building Dataset...')
        fpath = f"data/{dataset}.smi"
        with open(fpath, 'r') as fin:
            lines = fin.read().strip().split('\n')
        original_mols, original_logps, original_smis = [], [], []
        for line in lines:
            smi, logp = line.split()
            if 'Na' not in smi:
                original_smis.append(smi)
                original_logps.append(float(logp))
        
        print(len(original_smis), len(set(original_smis)))
        original_smis = list(set(original_smis))
        original_mols = parallel(smiles2molecule, original_smis, cpus)

        graphs = [predictor.convert.MolToGraph(m) for m in original_mols]
        data_loader = DataLoader(graphs, batch_size=128)
        predictions = []
        for batch in data_loader:
            preds, _ = predictor.predict(batch, target_class=0)
            predictions.extend(preds)

        negatives = [(i, m) for i, (m, p) in enumerate(zip(original_mols, predictions)) if p >= 0.5]
        negative_idx, negative_mols = zip(*negatives)

        preprocess_data = {
            'original_smis': original_smis,
            'predictions': predictions,
            'original_mols': original_mols,
            'negative_idx': negative_idx,
            'negative_mols': negative_mols
        }
        torch.save(preprocess_data, f'data/{dataset}.pt')

    for key in preprocess_data:
        print(key, len(preprocess_data[key]))


    results = []
    filename = configs['save_file']

    # Check if the file exists and delete it if it does
    if os.path.exists(filename):
        os.remove(filename)

    fout = open(filename, 'w')
    for i in tqdm(negative_idx, total=len(negative_idx)):
        print('='*50 + 'For Mol {}'.format(i) + '='*50)
        ori_mol, ori_smi = original_mols[i], original_smis[i]
        paths = annealing_sample(vae_model, predictor, ori_mol, ori_smi, negative_mols, configs)
        if paths == []:
            continue
        for x in paths:
            print(x[-1], x[-2], x[-3])
        print('='*50 + 'Mol {} done'.format(i) + '='*50)

        fout.write('{}\n'.format(json.dumps({ori_smi: [(x[-1], x[-2], x[-3]) for x in paths]})))
        fout.flush()

        # if DEBUG:
        #     if i > 10:
        #         break

    
        

    
    

