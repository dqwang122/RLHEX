
import math
import copy
import rdkit
import torch
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from rdkit import RDLogger
from collections import defaultdict

import torch
import torch.nn.functional as F
from torch_geometric.data import Data




# add edges, not complete
BOND_TYPES = {
    Chem.rdchem.BondType.SINGLE: 0,
    Chem.rdchem.BondType.DOUBLE: 1,
    Chem.rdchem.BondType.TRIPLE: 2,
    Chem.rdchem.BondType.AROMATIC: 3,
    None: -1}

BOND_ID_TO_TYPE = {v: k for k, v in BOND_TYPES.items()}


class Convertor():
    def __init__(self, keep_node_mapping):
        self.node_mapping = keep_node_mapping
        self.atom_types = {x:i for i,x in enumerate(keep_node_mapping)}

    def MolToGraph(self, mol):
        atom_types = self.atom_types
        g = Data()

        # add nodes
        def zinc_nodes(m):
            atom_feats_dict = defaultdict(list)
            num_atoms = m.GetNumAtoms()
            for u in range(num_atoms):
                atom = m.GetAtomWithIdx(u)
                symbol = atom.GetSymbol()
                if symbol not in atom_types:
                    continue
                atom_feats_dict['node_type'].append(atom_types[symbol])
            return atom_feats_dict

        atom_feats = zinc_nodes(mol)
        x = torch.LongTensor(atom_feats['node_type'])
        g.x = F.one_hot(x, num_classes=len(atom_types)).float()
        g.num_nodes = len(x)
        
        def zinc_edges(mol, edges, self_loop=False):
            bond_feats_dict = defaultdict(list)
            edges = [idxs.tolist() for idxs in edges]
            for e in range(len(edges[0])):
                u, v = edges[0][e], edges[1][e]
                if u == v and not self_loop: continue
                if mol.GetAtomWithIdx(u).GetSymbol() not in atom_types:
                    continue
                if mol.GetAtomWithIdx(v).GetSymbol() not in atom_types:
                    continue

                e_uv = mol.GetBondBetweenAtoms(u, v)
                if e_uv is None: bond_type = None
                else: bond_type = e_uv.GetBondType()
                bond_feats_dict['e_feat'].append(BOND_TYPES[bond_type])
            bond_feats_dict['e_feat'] = torch.LongTensor(bond_feats_dict['e_feat'])
            return bond_feats_dict
        
        edge_index = []
        bond_feats = []
        for bond in mol.GetBonds():
            u = bond.GetBeginAtomIdx()
            v = bond.GetEndAtomIdx()
            edge_index.append([u,v])
            edge_index.append([v,u])
        edge_index = sorted(edge_index, key=lambda x:x[0])
        edge_index = torch.LongTensor(edge_index).T
        bond_feats = zinc_edges(mol, edge_index)
        g.edge_index = edge_index
        g.edge_attr = bond_feats['e_feat']

        if g.x.shape[0] != mol.GetNumAtoms():
            print("Unknown atom type!")
            return None
        
        return g



    def MolFromGraph(self, Graph):
        keep_node_mapping = self.node_mapping

        # create empty editable mol object
        mol = Chem.RWMol()

        node_list = Graph.x             # [node_num, node_type]
        edge_index = Graph.edge_index.T # [edge_num, 2]
        edge_attr = Graph.edge_attr     # [edge_num]


        # add atoms to mol and keep track of index
        node_to_idx = {}
        for i in range(len(node_list)):
            node_cls_idx = torch.argmax(node_list[i]).item()
            node = keep_node_mapping[node_cls_idx]
            # print(node_cls_idx, node)
            a = Chem.Atom(node)
            molIdx = mol.AddAtom(a)
            node_to_idx[i] = molIdx

        # add bonds between adjacent atoms
        existing_edges = []
        for idx in range(edge_index.size(0)):
            edge = edge_index[idx]
            ix, iy = edge[0].item(), edge[1].item()
            if (iy, ix) in existing_edges:
                continue
            else:
                existing_edges.append((ix, iy))

            if edge_attr is not None:
                bond_type_id = edge_attr[idx].cpu().item()
                bond_type = BOND_ID_TO_TYPE[bond_type_id]
                mol.AddBond(node_to_idx[ix], node_to_idx[iy], bond_type)
            else:
                mol.AddBond(node_to_idx[ix], node_to_idx[iy])
        # Convert RWMol to Mol object
        mol = mol.GetMol()            

        return mol

    def valid_checking(self, g):
        try:
            if len(g.edge_attr) == g.edge_index.size(1):
                m = self.MolFromGraph(g)
                m.UpdatePropertyCache()
                Chem.SanitizeMol(m)
                problems = Chem.DetectChemistryProblems(m)
                if len(problems) == 0:
                    return True
        except:
            print(g.edge_attr)
            print(g.edge_attr is None)
        return False

def check_aromatic(mol):
    for bond in mol.GetBonds():
        if bond.GetBondType() == Chem.rdchem.BondType.AROMATIC:
            return True
    return False