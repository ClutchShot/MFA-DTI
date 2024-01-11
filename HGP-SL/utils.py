from rdkit import Chem
from dgllife.utils import mol_to_bigraph, PretrainAtomFeaturizer, PretrainBondFeaturizer
from torch_geometric.data import Data
import torch
import traceback


def read(filename: str) -> list:
    smiles = set()
    with open(f'data/dti/{filename}.txt', 'r') as reader:
        for line in reader:
            data = line.strip().split(' ')
            smiles.add(data[2])
    return list(smiles)

def read_amino(filename: str) -> list:
    smile = []
    amino_name = []
    with open(f'data/protein/{filename}.txt', 'r') as reader:
        for line in reader:
            data = line.strip().split(' ')
            if data[2] == 'None':
                continue
            amino_name.append(data[1])
            smile.append(data[2])
    return amino_name, smile


def graph_construction_and_featurization(smiles:list):
    graphs = []
    for smi in smiles:
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                print("NOT processed")
                continue
            g = mol_to_bigraph(mol, add_self_loop=True,
                               node_featurizer=PretrainAtomFeaturizer(),
                               edge_featurizer=PretrainBondFeaturizer(),
                               canonical_atom_order=False)
            u, v = g.edges()
            x = torch.unsqueeze(g.ndata['atomic_number'], 1).type(torch.float32)
            # u = u.to(torch.long)
            # v = v.to(torch.long)
            # edges = torch.stack([u, v], dim=0)
            edges = torch.stack([u, v], dim=0)
            edges = edges.to(torch.int64)
            y = torch.Tensor([1]).to(torch.int64)
            graphs.append(Data(x = x, edge_index = edges,y=y))
        except Exception:
            print("NOT processed")
            traceback.print_exc()


    return graphs

