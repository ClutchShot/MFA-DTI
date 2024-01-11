import numpy as np
from numpy.random import shuffle

def read_dti(dti_file:str, drugbank=False):
    dti = []
    with open(dti_file, 'r') as reader:
        for line in reader:
            data = line.strip().split(" ")
            if drugbank:
                dti.append(data[2:])
            else:
                dti.append(data)

    return dti

def to_vec(dti:list, vectors_dict:dict, flag:int):
    vectors = []
    for entity in dti:
        vectors.append(vectors_dict[entity[flag]])
    return np.array(vectors)

def read_proteins(filename: str) -> list:
    proteins = set()
    with open(f'data/dti/{filename}.txt', 'r') as reader:
        for line in reader:
            data = line.strip().split(' ')
            proteins.add(data[1])
    return list(proteins)