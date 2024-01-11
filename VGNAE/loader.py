import torch
import numpy as np
import scipy.sparse as sp
from torch_geometric.data import Data
from torch_geometric.utils import remove_self_loops, to_undirected
import math

class HashMap:

    def __init__(self) -> None:
        self.__dict__ = dict()
        self.__sizeof__ = 0

    def add(self, element:str):
        self.__dict__[element] = self.__sizeof__
        self.__sizeof__ += 1

    def get(self, element:str):
        return self.__dict__[element]
    
    def add_with_value(self, element:str, value:int):
        self.__dict__[element] = value

    def exist_element(self,element:str):
        return element in self.__dict__
    

class Loader:

    def __init__(self, fileName: str) -> None:
        self.graph = []
        self.node = set()
        self.drug = set()
        self.protein = set()
        self.__adj_matrics = None
        self.class_list = None
        self.dti_entities = set()
        self.__adj_matric_dti = None
        self.__dti = None
        
        self.load_not_labeles(fileName)

        self.__adj_matrics = np.zeros((len(self.node), len(self.node)))
        self.class_list = np.zeros((len(self.node),),dtype=int)

        self.label_matrics_nodes(flag=False)

        self.__adj_matric_dti = [self.__adj_matrics[i] for i in self.dti_entities]
        self.__adj_matric_dti = np.array(self.__adj_matric_dti, dtype=int)


    def process(self):
        x = sp.csr_matrix(self.__adj_matrics).todense()
        # x = sp.csr_matrix(self.__adj_matric_dti).todense()
        x = torch.from_numpy(x).to(torch.float)

        adj = sp.csr_matrix(self.__adj_matrics).tocoo()
        # adj = sp.csr_matrix(self.__adj_matric_dti).tocoo() 
        edge_index = torch.tensor([adj.col, adj.row], dtype=torch.long)

        edge_index, _ = remove_self_loops(edge_index)
        edge_index = to_undirected(edge_index, x.size(0)) 
        # edge_index = to_undirected(edge_index, x.size(1)) 

        y = torch.from_numpy(self.class_list).to(torch.long)

        return Data(x=x, edge_index=edge_index, y=y)
    

    def load_file(self, file_name:str ):
        with open(f"data/{file_name}.txt", 'r') as reader:
            for line in reader:
                data = line.strip().split(' ')
                pair_left = int(data[0])
                pair_right = int(data[1])
                interaction = int(data[2])

                self.graph.append([pair_left, pair_right, interaction])

                self.node.add(pair_left)
                self.node.add(pair_right)
    
    def load_not_labeles(self, file_name:str):
        hashmap = HashMap()
        pairs = []
        with open(f"data/dti/{file_name}.txt", 'r') as reader:
            for line in reader:
                data = line.strip().split(' ')
                pair_left = str(data[0])
                pair_right = str(data[1])
                if not hashmap.exist_element(pair_left):
                    hashmap.add(pair_left)
                if not hashmap.exist_element(pair_right):
                    hashmap.add(pair_right)
                interaction = int(data[2])
                pairs.append([pair_left,pair_right, interaction])

        for pair in pairs:
            left = hashmap.get(pair[0])
            right = hashmap.get(pair[1])
            self.graph.append([left, right, pair[2]])
            self.node.add(left)
            self.node.add(right)
            

    def label_matrics_nodes(self, flag:bool):
        if flag:
            for left, right, inter in self.graph:
                self.__adj_matrics[left][right] = inter
                # DTI
                if inter == 1:
                    self.class_list[left] = 0
                    self.class_list[right] = 1
                    self.dti_entities.add(left)
                    self.dti_entities.add(right)
                # DDI
                if inter == 2:
                    self.class_list[left] = 0
                    self.class_list[right] = 0
                # PPI 
                if inter == 3:
                    self.class_list[left] = 1
                    self.class_list[right] = 1
                # D-Side
                if inter == 4:
                    self.class_list[right] = 2
                # D-Disease
                if inter == 5:
                    self.class_list[right] = 3  
                # P-Disease
                if inter == 6:
                    self.class_list[right] = 3 
        else:
            for left, right, inter in self.graph:
                self.__adj_matrics[left][right] = inter
                self.class_list[left] = 0
                self.class_list[right] = 1
