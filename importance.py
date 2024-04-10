import torch

import numpy as np

import networkx as nx
from rdkit import Chem, DataStructs
from rdkit.Chem.QED import qed
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors
from rdkit.Chem import RDConfig

from torch_geometric.data import Data, Batch, DataLoader


from gnn import load_trained_gnn
from tools import  Convertor

def mol_similarity(mol1, mol2):
    try:
        # mol1.UpdatePropertyCache()
        # mol2.UpdatePropertyCache()
        # Chem.SanitizeMol(mol1)
        # Chem.SanitizeMol(mol2)
        fps1 = AllChem.GetMorganFingerprint(mol1, 2)
        fps2 = AllChem.GetMorganFingerprint(mol2, 2)
        return DataStructs.TanimotoSimilarity(fps1, fps2)
    except Exception as e:
        # print(e)
        try:
            mol1.UpdatePropertyCache()
            mol2.UpdatePropertyCache()
            Chem.SanitizeMol(mol1)
            Chem.SanitizeMol(mol2)
            fps1 = AllChem.GetMorganFingerprint(mol1, 2)
            fps2 = AllChem.GetMorganFingerprint(mol2, 2)
            return DataStructs.TanimotoSimilarity(fps1, fps2)
        except Exception as e:
            return 0
        # return 0

def mol_distance(mol1, mol2, norm=False):
    dist = 1 - mol_similarity(mol1, mol2)
    if norm:
        mol1_size = mol1.GetNumAtoms() + mol1.GetNumBonds()
        mol2_size = mol2.GetNumAtoms() + mol2.GetNumBonds()
        dist /= (mol1_size + mol2_size)

    return dist

def mol_logp(mol):
    return Descriptors.MolLogP(mol)

class GNNPredictor():
    def __init__(self, gnn_ckpt, dataset_name, keep_node_mapping, device):
        self.device = device
        self.model = load_trained_gnn(dataset_name, gnn_ckpt, device=device)
        self.model.eval()
        self.convert = Convertor(keep_node_mapping)
    
    def predict(self, graphs, target_class=1):
        graphs = graphs.to(self.device)
        node_embeddings, graph_embeddings, preds = self.model(graphs)  # .to(model.device))
        preds = torch.exp(preds)
        return preds[:, target_class], graph_embeddings

    def get_graph_represetation(self, graphs):
        graphs = graphs.to(self.device)
        _, graph_embeddings, _ = self.model(graphs)
        return graph_embeddings
    
    def get_node_represetation(self, graphs):
        graphs = graphs.to(self.device)
        node_embeddings, _, _ = self.model(graphs)
        return node_embeddings
    
    def predict_mol(self, mol, target_class=1):
        graphs = self.convert.MolToGraph(mol)
        return self.predict(graphs, target_class=target_class)
    
class Distance():
    def __init__(self, original_mols, threshold=0.05, dist_func=mol_distance, norm=False):
        self.original_mols = original_mols
        self.dist_func = dist_func
        self.threshold = threshold
        self.norm = norm
    
    def coverage(self, mol, penalty=False):
        """
        coverage:
        proportion of molecules in the original dataset 
        that are within the threshold distance
        """
        dists = [self.dist_func(mol, m, self.norm) for m in self.original_mols]
        coverage = sum([d <= self.threshold for d in dists]) / len(self.original_mols)
        return coverage, dists

class Importance():
    def __init__(self, original_mols, predictor, threshold=0.05, norm=False, dist_func=mol_distance):
        self.gnn_predictor = predictor
        self.threshold = threshold
        self.dist_model = Distance(original_mols, threshold=threshold, dist_func=dist_func, norm=norm)

        print(predictor)
    
    def get_importance(self, mol, target_class=1, keys=['coverage', 'prediction'], weights=[1, 1]):
        assert len(keys) == len(weights)

        results = {}
        if 'prediction' in keys:
            graph = self.gnn_predictor.convert.MolToGraph(mol)
            pred, _ = self.gnn_predictor.predict(graph, target_class=target_class)
            results['prediction'] = pred.item()
        if 'coverage' in keys:
            coverage, dists = self.dist_model.coverage(mol)
            results['coverage'] = coverage
        else:
            dists = None
        importance = sum([results[k] * w for k, w in zip(keys, weights)])

        return importance, results, dists
    

    def get_importance_batch(self, mols, batch_size=32, 
                            target_class=1, keys=['coverage', 'prediction'],
                            weights=[1, 1], **kwargs):
        assert len(keys) == len(weights)

        prediction_threshold = 0.5
        coverage_threshold = 0.05
        
        results = {k: [] for k in keys}
        for batch_mol in [mols[i:i+batch_size] for i in range(0, len(mols), batch_size)]:
            # batch_small_mol = [m for m in batch_mol if m.GetNumAtoms() <= 60 and m.GetNumBonds() >= 2]
            batch_small_mol = [m for m in batch_mol if m is not None and m.GetNumBonds() >= 0 and m.GetNumBonds() >= 2 and m.GetNumAtoms() <= 100]
            # print('small mol', [m.GetNumBonds() for m in batch_small_mol])
            # batch_small_mol = batch_mol
            if len(batch_small_mol) == 0:
                for k in keys:
                    results[k].extend([0] * len(batch_small_mol))
                continue
                
            if 'prediction' in keys:
                prediction = [0] * len(batch_small_mol)
                graphs = [self.gnn_predictor.convert.MolToGraph(m) for m in batch_small_mol]
                if len([g for g in graphs if g is not None]) == 0:
                    for k in keys:
                        results[k].extend([0] * len(batch_small_mol))
                    continue
                graph_idx, graphs = zip(*[(i,g) for i, g in enumerate(graphs) if g is not None])
                batch_graph = Batch.from_data_list(graphs)
                # print('graphs', batch_graph) 
                preds, _ = self.gnn_predictor.predict(batch_graph, target_class=target_class)
                for i, p in zip(graph_idx, preds):
                    prediction[i] = p.item()
                print(f"raw_prediction: {prediction}")
                if kwargs.get('binary_reward', False):
                    prediction = [int(p >= 0.5) for p in prediction]
                elif kwargs.get('half_reward', False):
                    prediction = [p-0.5 for p in prediction]
                print('prediction', prediction)
                results['prediction'].extend(prediction)
            if 'coverage' in keys:
                ret = [self.dist_model.coverage(m) if m is not None else (0, []) for m in batch_small_mol]
                results['coverage'].extend([r[0] for r in ret])
            

            if kwargs.get('debug', False):
                print("="*20)
                for mol in batch_mol:
                    print("-//-"*20)
                    print(f"Molecule: {mol}")
                    for key in keys:
                        print(f"Reward key: {key}, Unweighted Value: {results[key][-1]}")
                    if kwargs.get('ratio_weight', False):
                        print(f"RATIO WEIGHT: {kwargs.get('ratio_weight')}")
                        assert len(keys) == 2, "Ratio weight requires exactly two keys"
                        if results[keys[1]][-1] != 0:
                            ratio_weight_value = results[keys[0]][-1] + ((results[keys[0]][-1] / results[keys[1]][-1]) * results[keys[1]][-1])
                        else:
                            ratio_weight_value = 0
                        print(f"Ratio Weight Value: {ratio_weight_value}")
                    else:
                        for key in keys:
                            print(f"Weighted ({weights[keys.index(key)]}) Value: {results[key][-1] * weights[keys.index(key)]}")
                print("="*20)
        print(f"weight_ratio: {kwargs.get('ratio_weight')}")

        # for k, v in results.items():
        #     print(k, len(v), v)
        if kwargs.get('ratio_weight', False):
            # Ensure there are exactly two keys for ratio calculation
            assert len(keys) == 2, "Ratio weight requires exactly two keys"
            importance = []
            for i in range(len(results[keys[0]])):
                if kwargs.get('debug', False):
                    print(f"coverage: {results[keys[0]][i]}, prediction: {results[keys[1]][i]}")
                if results[keys[0]][i] != 0:
                    ratio = (results[keys[1]][i] / results[keys[0]][i])
                    imp = results[keys[1]][i] + (ratio * results[keys[0]][i])
                else:
                    imp = 0 - kwargs.get('penalty', 0)
                importance.append(imp)
        else:
            # results = {k: v for k, v in results.items() if k != 'prediction'}
            print(f"results: {results}, weights: {weights}")
            importance = []
            for i in range(len(results[keys[0]])):
                cov_rew = results[keys[0]][i]
                pred_rew = results[keys[1]][i]
                cov_rew_w = cov_rew * weights[0]
                pred_rew_w = pred_rew * weights[1]
                if kwargs.get('bonus_for_both', False):
                    assert kwargs.get('binary_reward', False) == True, "bonus_for_both requires binary_reward"
                    print(f"Setting bonus for both")
                    if cov_rew > coverage_threshold and pred_rew > prediction_threshold:
                        print(f"FOUND BONUS FOR BOTH!!!!!!!!")
                        imp = cov_rew_w + pred_rew_w    # add *weighted* rewards
                        print(f"Importance: {imp}")
                    else:
                        imp = -0.1
                else:
                    imp = cov_rew_w + pred_rew_w
                
                # if kwargs.get('debug', False):
                #     print(f"coverage: {cov_rew}, prediction: {pred_rew}")
                #     print(f"cov_rew_w: {cov_rew_w}, pred_rew_w: {pred_rew_w}")
                #     print(f"Importance: {imp}")
                #     # if idx > 1:
                #     #     break
                #     # wait for user to press a key to continue
                #     print(f"HIIII")
                #     input("Press Enter to continue...")

                print(f"Importance: {imp}")
                importance.append(imp)
        
        results_dict = []
        for i in range(len(results[keys[0]])):
            result = {}
            for k in keys:
                result[k] = results[k][i]
            results_dict.append(result)

        if len(importance) == 0:
            importance = [0] * len(mols)
        return importance, results_dict

    def get_importance_batch_2(self, mols, batch_size=32, target_class=1, keys=['coverage', 'prediction'], weights=[1, 1], **kwargs):
        assert len(keys) == len(weights)

        results = []
        for batch_mol in [mols[i:i+batch_size] for i in range(0, len(mols), batch_size)]:
            batch_results = {k: [0] * batch_size for k in keys}
            batch_small_mol = [m for m in batch_mol if m.GetNumAtoms() <= 50]
            if len(batch_small_mol) == 0:
                results.append(batch_results)
                continue
            if 'prediction' in keys:
                graphs = [self.gnn_predictor.convert.MolToGraph(m) for m in batch_mol]
                batch_graph = Batch.from_data_list(graphs)
                preds, _ = self.gnn_predictor.predict(batch_graph, target_class=target_class)
                batch_results['prediction'] = preds[0].item()
            if 'coverage' in keys:
                ret = [self.dist_model.coverage(m, **kwargs) for m in batch_mol]
                batch_results['coverage'] = ret[0][0]
            results.append(batch_results)

        importance = [sum([result[k] * w for k, w in zip(keys, weights)]) for i in range(len(mols)) for result in results]

        return importance, results

    def get_importance_from_graph_batch(self, graphs, batch_size=32, target_class=1, keys=['coverage', 'prediction'], weights=[1, 1]):
        assert len(keys) == len(weights)
        
        results = {k: [] for k in keys}
        for batch_graph in [graphs[i:i+batch_size] for i in range(0, len(graphs), batch_size)]:
            if 'prediction' in keys:
                batched_graph = Batch.from_data_list(batch_graph)
                preds, _ = self.gnn_predictor.predict(batched_graph, target_class=target_class)
                results['prediction'].extend([p.item() for p in preds])
            if 'coverage' in keys:
                batch_mol = [self.gnn_predictor.convert.GraphToMol(g) for g in batch_graph]
                ret = [self.dist_model.coverage(m) for m in batch_mol]
                results['coverage'].extend([r[0] for r in ret])
                        
        importance = [sum([results[k][i] * w for k, w in zip(keys, weights)]) for i in range(len(graphs))]

        return importance, results
    
    def get_matrix_importance_from_graph(self, graphs, batch_size=32, target_class=1, keys=['coverage', 'prediction'], weights=[1, 1]):
        assert len(keys) == len(weights)
        
        results = {k: [] for k in keys}
        graph_embeddings_list = []
        dist_list = []
        with torch.no_grad():
            for batch_graph in [graphs[i:i+batch_size] for i in range(0, len(graphs), batch_size)]:
                if 'prediction' in keys:
                    batched_graph = Batch.from_data_list(batch_graph)
                    preds, graph_embeddings = self.gnn_predictor.predict(batched_graph, target_class=target_class)
                    graph_embeddings = graph_embeddings.cpu().numpy()
                    graph_embeddings_list.extend(graph_embeddings)
                    results['prediction'].extend([p.item() for p in preds])
                if 'coverage' in keys:
                    batch_mol = [self.gnn_predictor.convert.MolFromGraph(g) for g in batch_graph]
                    ret = [self.dist_model.coverage(m) for m in batch_mol]
                    results['coverage'].extend([r[0] for r in ret])
                    dist_list.extend([r[1] for r in ret])
        graph_embeddings_matrix = np.array(graph_embeddings_list)
        dist_list = np.array(dist_list)
        coverage_matrix = (dist_list <= self.threshold).astype(int)

        # importance = [sum([results[k][i] * w for k, w in zip(keys, weights)]) for i in range(len(graphs))]

        score_matrix = np.stack([results['prediction'], results['coverage']]).T

        print(score_matrix.shape, graph_embeddings_matrix.shape, coverage_matrix.shape)
        coverage_matrix = torch.tensor(coverage_matrix)
        return score_matrix, graph_embeddings_matrix, coverage_matrix

        
        

