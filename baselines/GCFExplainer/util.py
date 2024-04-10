import torch
from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')   


def graph_element_counts(dataset):
    return torch.Tensor([graph.num_nodes + graph.num_edges / 2 for graph in dataset])

def th_delete(tensor, indices):
    mask = torch.ones(tensor.numel(), dtype=torch.bool)
    mask[indices] = False
    return tensor[mask]

def MolFromGraphs(Graph, keep_node_mapping):

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
            bond_type_id = edge_attr[idx]
            if bond_type_id == 0:
                bond_type = Chem.rdchem.BondType.SINGLE
            elif bond_type_id == 1:
                bond_type = Chem.rdchem.BondType.DOUBLE
            elif bond_type_id == 2:
                bond_type = Chem.rdchem.BondType.TRIPLE
            elif bond_type_id == 3:
                bond_type = Chem.rdchem.BondType.AROMATIC
            else:
                bond_type = None
            mol.AddBond(node_to_idx[ix], node_to_idx[iy], bond_type)
        else:
            mol.AddBond(node_to_idx[ix], node_to_idx[iy])
    # Convert RWMol to Mol object
    mol = mol.GetMol()            

    return mol

def valid_checking(g, keep_node_mapping):
    try:
        if len(g.edge_attr) == g.edge_index.size(1):
            m = MolFromGraphs(g, keep_node_mapping)
            m.UpdatePropertyCache()
            Chem.SanitizeMol(m)
            problems = Chem.DetectChemistryProblems(m)
            if len(problems) == 0:
                return True
        else:
            # print(g.edge_attr is None)
            print(f"edge_attr: {len(g.edge_attr)}, edge_index: {g.edge_index.size(1)}")
            return False
    except:
        # print(g.edge_attr is None)
        return False
