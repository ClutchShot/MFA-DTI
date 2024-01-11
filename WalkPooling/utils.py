from torch_geometric.data import Data
from torch_geometric.loader import DataLoader, DataListLoader
import torch
import argparse
import numpy as np
from numpy.random import shuffle
import math
from torch_geometric.utils import to_undirected, from_scipy_sparse_matrix,dense_to_sparse,is_undirected
import torch.nn.functional as F
import sys
import os.path
from loader import Loader


device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')

def floor(x):
    return torch.div(x, 1, rounding_mode='trunc')

def set_random_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def split_edges_new(data,args):
    set_random_seed(args.seed)
    row, col = data.edge_index
    mask = row < col
    row, col = row[mask], col[mask]
    n_v= floor(args.val_ratio * row.size(0)).int() #number of validation positive edges
    n_t=floor(args.test_ratio * row.size(0)).int() #number of test positive edges
    #split positive edges   
    perm = torch.randperm(row.size(0))
    row, col = row[perm], col[perm]
    r, c = row[:n_v], col[:n_v]
    data.val_pos = torch.stack([r, c], dim=0)
    r, c = row[n_v:n_v+n_t], col[n_v:n_v+n_t]
    data.test_pos = torch.stack([r, c], dim=0)
    r, c = row[n_v+n_t:], col[n_v+n_t:]
    train_pos = torch.stack([r, c], dim=0)
    data.train_pos = torch.stack([r, c], dim=0)

    # Sample the test negative edges first
    neg_row, neg_col =  data.edge_index_neg
    mask_neg = neg_row < neg_col

    neg_row, neg_col = neg_row[mask_neg], neg_col[mask_neg]
    perm = torch.randperm(neg_row.size(0))[:n_t]
    row, col = neg_row[perm], neg_col[perm]
    data.test_neg = torch.stack([row, col], dim=0)

    n_tot = n_v + data.train_pos.size(1)
    perm = torch.randperm(neg_row.size(0))
    neg_row, neg_col = neg_row[perm], neg_col[perm]

    row, col = neg_row[:n_v], neg_col[:n_v]
    data.val_neg = torch.stack([row, col], dim=0)

    row, col = neg_row[n_v:], neg_col[n_v:]
    data.train_neg = torch.stack([row, col], dim=0)
    
    return data

def split_edges(data,args):
    set_random_seed(args.seed)
    row, col = data.edge_index
    mask = row < col
    row, col = row[mask], col[mask]
    n_v= floor(args.val_ratio * row.size(0)).int() #number of validation positive edges
    n_t=floor(args.test_ratio * row.size(0)).int() #number of test positive edges
    #split positive edges   
    # perm = torch.randperm(row.size(0))
    # row, col = row[perm], col[perm]
    r, c = row[:n_v], col[:n_v]
    data.val_pos = torch.stack([r, c], dim=0)
    r, c = row[n_v:n_v+n_t], col[n_v:n_v+n_t]
    data.test_pos = torch.stack([r, c], dim=0)
    r, c = row[n_v+n_t:], col[n_v+n_t:]
    data.train_pos = torch.stack([r, c], dim=0)

    #sample negative edges
    if args.practical_neg_sample == False:
        # If practical_neg_sample == False, the sampled negative edges
        # in the training and validation set aware the test set

        neg_adj_mask = torch.ones(data.num_nodes, data.num_nodes, dtype=torch.uint8)
        neg_adj_mask = neg_adj_mask.triu(diagonal=1).to(torch.bool)
        neg_adj_mask[row, col] = 0

        # Sample all the negative edges and split into val, test, train negs
        neg_row, neg_col = neg_adj_mask.nonzero(as_tuple=False).t()
        perm = torch.randperm(neg_row.size(0))[:row.size(0)]
        neg_row, neg_col = neg_row[perm], neg_col[perm]

        row, col = neg_row[:n_v], neg_col[:n_v]
        data.val_neg = torch.stack([row, col], dim=0)

        row, col = neg_row[n_v:n_v + n_t], neg_col[n_v:n_v + n_t]
        data.test_neg = torch.stack([row, col], dim=0)

        row, col = neg_row[n_v + n_t:], neg_col[n_v + n_t:]
        data.train_neg = torch.stack([row, col], dim=0)

    else:

        neg_adj_mask = torch.ones(data.num_nodes, data.num_nodes, dtype=torch.uint8)
        neg_adj_mask = neg_adj_mask.triu(diagonal=1).to(torch.bool)
        neg_adj_mask[row, col] = 0

        # Sample the test negative edges first
        neg_row, neg_col = neg_adj_mask.nonzero(as_tuple=False).t()
        perm = torch.randperm(neg_row.size(0))[:n_t]
        neg_row, neg_col = neg_row[perm], neg_col[perm]
        data.test_neg = torch.stack([neg_row, neg_col], dim=0)

        # Sample the train and val negative edges with only knowing 
        # the train positive edges
        row, col = data.train_pos
        neg_adj_mask = torch.ones(data.num_nodes, data.num_nodes, dtype=torch.uint8)
        neg_adj_mask = neg_adj_mask.triu(diagonal=1).to(torch.bool)
        neg_adj_mask[row, col] = 0

        # Sample the train and validation negative edges
        neg_row, neg_col = neg_adj_mask.nonzero(as_tuple=False).t()

        n_tot = n_v + data.train_pos.size(1)
        perm = torch.randperm(neg_row.size(0))[:n_tot]
        neg_row, neg_col = neg_row[perm], neg_col[perm]

        row, col = neg_row[:n_v], neg_col[:n_v]
        data.val_neg = torch.stack([row, col], dim=0)

        row, col = neg_row[n_v:], neg_col[n_v:]
        data.train_neg = torch.stack([row, col], dim=0)

    return data

def k_hop_subgraph(node_idx, num_hops, edge_index, max_nodes_per_hop = None,num_nodes = None):
  
    if num_nodes == None:
        num_nodes = torch.max(edge_index)+1
    row, col = edge_index
    node_mask = row.new_empty(num_nodes, dtype=torch.bool)
    edge_mask = row.new_empty(row.size(0), dtype=torch.bool)

    node_idx = node_idx.to(row.device)

    subsets = [node_idx]

    if max_nodes_per_hop == None:
        for _ in range(num_hops):
            node_mask.fill_(False)
            node_mask[subsets[-1]] = True
            torch.index_select(node_mask, 0, row, out = edge_mask)
            subsets.append(col[edge_mask])
    else:
        not_visited = row.new_empty(num_nodes, dtype=torch.bool)
        not_visited.fill_(True)
        for _ in range(num_hops):
            node_mask.fill_(False)# the source node mask in this hop
            node_mask[subsets[-1]] = True #mark the sources
            not_visited[subsets[-1]] = False # mark visited nodes
            torch.index_select(node_mask, 0, row, out = edge_mask) # indices of all neighbors
            neighbors = col[edge_mask].unique() #remove repeats
            neighbor_mask = row.new_empty(num_nodes, dtype=torch.bool) # mask of all neighbor nodes
            edge_mask_hop = row.new_empty(row.size(0), dtype=torch.bool) # selected neighbor mask in this hop
            neighbor_mask.fill_(False)
            neighbor_mask[neighbors] = True
            neighbor_mask = torch.logical_and(neighbor_mask, not_visited) # all neighbors that are not visited
            ind = torch.where(neighbor_mask==True) #indicies of all the unvisited neighbors
            if ind[0].size(0) > max_nodes_per_hop:
                perm = torch.randperm(ind[0].size(0))
                ind = ind[0][perm]
                neighbor_mask[ind[max_nodes_per_hop:]] = False # randomly select max_nodes_per_hop nodes
                torch.index_select(neighbor_mask, 0, col, out = edge_mask_hop)# find the indicies of selected nodes
                edge_mask = torch.logical_and(edge_mask,edge_mask_hop) # change edge_mask
            subsets.append(col[edge_mask])

    subset, inv = torch.cat(subsets).unique(return_inverse=True)
    inv = inv[:node_idx.numel()]

    node_mask.fill_(False)
    node_mask[subset] = True
    edge_mask = node_mask[row] & node_mask[col]

    edge_index = edge_index[:, edge_mask]

    node_idx = row.new_full((num_nodes, ), -1)
    node_idx[subset] = torch.arange(subset.size(0), device=row.device)
    edge_index = node_idx[edge_index]

    return subset, edge_index, inv, edge_mask

def plus_edge(data_observed, label, p_edge, args):
    nodes, edge_index_m, mapping, _ = k_hop_subgraph(node_idx= p_edge, num_hops=args.num_hops,\
 edge_index = data_observed.edge_index, max_nodes_per_hop=args.max_nodes_per_hop ,num_nodes=data_observed.num_nodes)
    x_sub = data_observed.x[nodes,:]
    edge_index_p = edge_index_m
    edge_index_p = torch.cat((edge_index_p, mapping.view(-1,1)),dim=1)
    edge_index_p = torch.cat((edge_index_p, mapping[[1,0]].view(-1,1)),dim=1)

    #edge_mask marks the edge under perturbation, i.e., the candidate edge for LP
    edge_mask = torch.ones(edge_index_p.size(1),dtype=torch.bool)
    edge_mask[-1] = False
    edge_mask[-2] = False

    data = Data(edge_index = edge_index_p, x = x_sub, z = 0)
    data.edge_mask = edge_mask

    data.label = float(label)

    return data

def minus_edge(data_observed, label, p_edge, args):
    nodes, edge_index_p, mapping,_ = k_hop_subgraph(node_idx= p_edge, num_hops=args.num_hops,\
 edge_index = data_observed.edge_index,max_nodes_per_hop=args.max_nodes_per_hop, num_nodes = data_observed.num_nodes)
    x_sub = data_observed.x[nodes,:]

    #edge_mask marks the edge under perturbation, i.e., the candidate edge for LP
    edge_mask = torch.ones(edge_index_p.size(1), dtype = torch.bool)
    ind = torch.where((edge_index_p == mapping.view(-1,1)).all(dim=0))
    edge_mask[ind[0]] = False
    ind = torch.where((edge_index_p == mapping[[1,0]].view(-1,1)).all(dim=0))
    edge_mask[ind[0]] = False
    data = Data(edge_index = edge_index_p, x= x_sub,z = 0)
    data.edge_mask = edge_mask

    data.label = float(label)
    return data


def load_splitted_data(args):
    par_dir = os.path.abspath(os.path.join(os.path.dirname(__file__),".."))
    data_name=args.data_name+'_split_'+args.data_split_num
    if args.test_ratio==0.5:
        data_dir = os.path.join(par_dir, 'data/splitted_0_5/{}.mat'.format(data_name))
    else:
        data_dir = os.path.join(par_dir, 'data/splitted/{}.mat'.format(data_name))
    import scipy.io as sio
    print('Load data from: '+data_dir)
    net = sio.loadmat(data_dir)
    data = Data()

    data.train_pos = torch.from_numpy(np.int64(net['train_pos']))
    data.train_neg = torch.from_numpy(np.int64(net['train_neg']))
    data.test_pos = torch.from_numpy(np.int64(net['test_pos']))
    data.test_neg = torch.from_numpy(np.int64(net['test_neg']))

    n_pos= floor(args.val_ratio * len(data.train_pos)).int()
    nlist=np.arange(len(data.train_pos))
    np.random.shuffle(nlist)
    val_pos_list=nlist[:n_pos]
    train_pos_list=nlist[n_pos:]
    data.val_pos=data.train_pos[val_pos_list]
    data.train_pos=data.train_pos[train_pos_list]

    n_neg = floor(args.val_ratio * len(data.train_neg)).int()
    nlist=np.arange(len(data.train_neg))
    np.random.shuffle(nlist)
    val_neg_list=nlist[:n_neg]
    train_neg_list=nlist[n_neg:]
    data.val_neg=data.train_neg[val_neg_list]
    data.train_neg=data.train_neg[train_neg_list]

    data.val_pos = torch.transpose(data.val_pos,0,1)
    data.val_neg = torch.transpose(data.val_neg,0,1)
    data.train_pos = torch.transpose(data.train_pos,0,1)
    data.train_neg = torch.transpose(data.train_neg,0,1)
    data.test_pos = torch.transpose(data.test_pos,0,1)
    data.test_neg = torch.transpose(data.test_neg,0,1)
    num_nodes = max(torch.max(data.train_pos), torch.max(data.test_pos))+1
    num_nodes=max(num_nodes,torch.max(data.val_pos)+1)
    data.num_nodes = num_nodes

    return data



def set_init_attribute_representation(data,args):
    #Construct data_observed and compute its node attributes & representation
    edge_index = torch.cat((data.train_pos,data.train_pos[[1,0],:]),dim=1)
    if args.observe_val_and_injection == False:
        data_observed = Data(edge_index=edge_index)
    else:
        edge_index=torch.cat((edge_index,data.val_pos,data.val_pos[[1,0],:]),dim=1)
        data_observed = Data(edge_index=edge_index)
    data_observed.num_nodes = data.num_nodes
    if args.observe_val_and_injection == False:
        edge_index_observed = data_observed.edge_index
    else: 
        #use the injection trick and add val data in observed graph 
        edge_index_observed = torch.cat((data_observed.edge_index,\
            data.train_neg,data.train_neg[[1,0],:],data.val_neg,data.val_neg[[1,0],:]),dim=1)

    if data.x != None:
        x = data.x

    if args.init_representation == 'VGNAE':
        data_observed.x = x
        feature_results=None

    return data_observed,feature_results

def prepare_data(args):
    datasets =  ['drugbank', 'human_v1', 'celegans', 'GPCR', 'BindingDB', 'Davis']

    if args.data_name in datasets:
        loader = Loader(args.data_name)
        data = loader.process()
        x_len = data.x.size(0)
        feature_vectors = np.load(f"data/dti_vectors/{args.data_name}_128.npy")
        feature_vectors = feature_vectors[:x_len]
        feature_tensor = torch.from_numpy(feature_vectors)
        data.x = feature_tensor
        # Deafault
        # data = split_edges(data,args)
        # NEG POS split
        data = split_edges_new(data,args)

    set_random_seed(args.seed)
    data_observed,feature_results = set_init_attribute_representation(data,args)

    #Construct train, val and test data loader.
    set_random_seed(args.seed)
    train_graphs = []
    val_graphs = []
    test_graphs = []
    vector_graph = []

    for i in range(data.train_pos.size(1)):
        train_graphs.append(minus_edge(data_observed,1,data.train_pos[:,i],args))

    for i in range(data.train_neg.size(1)):
        train_graphs.append(plus_edge(data_observed,0,data.train_neg[:,i],args))

    for i in range(data.test_pos.size(1)):
        test_graphs.append(plus_edge(data_observed,1,data.test_pos[:,i],args))

    for i in range(data.test_neg.size(1)):
        test_graphs.append(plus_edge(data_observed,0,data.test_neg[:,i],args))   
    if args.observe_val_and_injection == False:
        for i in range(data.val_pos.size(1)):
            val_graphs.append(plus_edge(data_observed,1,data.val_pos[:,i],args))

        for i in range(data.val_neg.size(1)):
            val_graphs.append(plus_edge(data_observed,0,data.val_neg[:,i],args))
    else:
        for i in range(data.val_pos.size(1)):
            val_graphs.append(minus_edge(data_observed,1,data.val_pos[:,i],args))

        for i in range(data.val_neg.size(1)):
            val_graphs.append(plus_edge(data_observed,0,data.val_neg[:,i],args))


    for i in range(data.masks.shape[0]):
        pair = torch.tensor(data.masks[i][:2], dtype=torch.int64)
        # vector_graph.append(plus_edge(data_observed,data.masks[i][2],data.masks[i][:2],args))
        vector_graph.append(plus_edge(data_observed, data.masks[i][2], pair, args))
    
    
    print('Train_link:', str(len(train_graphs)),' Val_link:',str(len(val_graphs)),' Test_link:',str(len(test_graphs)), 'Get vectors', str(len(vector_graph)))


    if args.multi_gpu:
        # DataParallel torch_geometric
        train_loader = DataListLoader(train_graphs,batch_size=args.batch_size,shuffle=True, drop_last=True)
        val_loader = DataListLoader(val_graphs,batch_size=args.batch_size,shuffle=True, drop_last=True)
        test_loader = DataListLoader(test_graphs,batch_size=args.batch_size,shuffle=False, drop_last=True)
        vectors_loader = DataListLoader(vector_graph,batch_size=args.batch_size,shuffle=False)
    else:
        # Single GPU
        train_loader = DataLoader(train_graphs,batch_size=args.batch_size,shuffle=True, drop_last=True)
        val_loader = DataLoader(val_graphs,batch_size=args.batch_size,shuffle=True, drop_last=True)
        test_loader = DataLoader(test_graphs,batch_size=args.batch_size,shuffle=False, drop_last=True)
        vectors_loader = DataLoader(vector_graph,batch_size=args.batch_size,shuffle=False)

    return train_loader, val_loader, test_loader,feature_results, vectors_loader


class EarlyStopping:
    def __init__(self, patience=1, min_delta=0.1):

        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def __call__(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False
