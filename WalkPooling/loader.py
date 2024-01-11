import torch
import numpy as np
from numpy.random import shuffle
import scipy.sparse as sp
from torch_geometric.data import Data
from torch_geometric.utils import remove_self_loops, to_undirected
import math

class HashMap:

    def __init__(self) -> None:
        self.dict = dict()
        self.size = 0

    def add(self, element:str):
        self.dict[element] = self.size
        self.size +=1

    def get(self, element:str):
        return self.dict[element]
    
    def add_with_value(self, element:str, value:int):
        self.dict[element] = value

    def exist_element(self,element:str):
        return element in self.dict
    
    def size_dict(self):
        return len(self.dict)
    
    def reverce_dict(self):
        return {v: k for k, v in self.dict.items()}
    

class Loader:

    def __init__(self, fileName: str) -> None:
        self.graph = []
        self.node = set()
        self.__adj_matrics = None
        self.class_list = None
        self.hashmap = None
        self.left_pos = []
        self.right_pos = []
        self.left_neg = []
        self.right_neg = []
        self.mask_indexes_pos = []
        self.mask_indexes_neg = []
        
        self.load_not_labeles(fileName)

        self.__adj_matrics = np.zeros((len(self.node), len(self.node)), dtype=int)
        self.__adj_matrics_neg = np.zeros((len(self.node), len(self.node)), dtype=int)
        self.class_list = np.zeros((len(self.node),),dtype=int)

        self.label_matrics_nodes()



    def process(self):
        x = sp.csr_matrix(self.__adj_matrics).todense()
        x = torch.from_numpy(x).to(torch.float)

        adj = sp.csr_matrix(self.__adj_matrics).tocoo()
        adj_neg = sp.csr_matrix(self.__adj_matrics_neg).tocoo()
        edge_index_pos = torch.tensor([adj.col, adj.row], dtype=torch.long)
        edge_index_neg = torch.tensor([adj_neg.col, adj_neg.row], dtype=torch.long)
        
        edge_index_pos, _ = remove_self_loops(edge_index_pos)
        edge_index_pos = to_undirected(edge_index_pos, x.size(0)) 

        edge_index_neg, _ = remove_self_loops(edge_index_neg)
        edge_index_neg = to_undirected(edge_index_neg, x.size(0)) 

        y = torch.from_numpy(self.class_list).to(torch.long)

        data = Data(x=x, edge_index=edge_index_pos, y=y)
        data.edge_index_neg = edge_index_neg

        data.masks = torch.from_numpy(np.array(self.graph))
        data.adj_matrics = self.__adj_matrics

        return data
    

    def load_not_labeles(self, file_name:str):
        self.hashmap = HashMap()
        pairs = []
        with open(f"data/dti/{file_name}.txt", 'r') as reader:
            for line in reader:
                data = line.strip().split(' ')
                pair_left = str(data[0])
                pair_right = str(data[1])
                if not self.hashmap.exist_element(pair_left):
                    self.hashmap.add(pair_left)
                if not self.hashmap.exist_element(pair_right):
                    self.hashmap.add(pair_right)
                interaction =  float(data[2])
                pairs.append([pair_left, pair_right, interaction])

        for pair in pairs:
            left = self.hashmap.get(pair[0])
            right = self.hashmap.get(pair[1])
            self.graph.append([left, right, pair[2]])
            self.node.add(left)
            self.node.add(right)


    def label_matrics_nodes(self):
        for index, data in enumerate(self.graph):
            left, right, inter = data
            self.__adj_matrics[left][right] = inter
            if inter == 0:
                self.__adj_matrics_neg[left][right] = 1
            # DTI
            if inter == 1:
                self.mask_indexes_pos.append(index)
                self.left_pos.append(left)
                self.right_pos.append(right)
            if inter == 0:
                self.mask_indexes_neg.append(index)
                self.left_neg.append(left)
                self.right_neg.append(right)


        