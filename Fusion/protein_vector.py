import numpy as np
from torch import nn
from torch.nn import functional as F
import torch
import math
from utlis import read_proteins
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='drugbank')
args = parser.parse_args()
DATASET = args.dataset

amino_vectors = np.load('data/amino/amino_acids_128.npy',allow_pickle=True).item()

proteins = read_proteins(DATASET)

proteins_vectors = []
vectors_full = []
for protein in proteins:
    p_vector = []
    not_contain = []
    for amino_char in protein:
        try:
            p_vector.append(amino_vectors[amino_char])
        except:
            not_contain.append(amino_char)

    vectors_full.append(len(p_vector))
    proteins_vectors.append(torch.from_numpy(np.array(p_vector)).unsqueeze(dim=0))


model = nn.AdaptiveMaxPool2d((1024,128))
dict_vectors = {}

for i in range(len(proteins)):
    out = model(proteins_vectors[i].unsqueeze(dim=0))
    vector = out.squeeze(dim=0).squeeze(dim=0)
    dict_vectors[proteins[i]] = vector.cpu().detach().numpy()

np.save(f"data/protein/{DATASET}_protein_1024_128_max", dict_vectors)
