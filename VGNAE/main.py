import os.path as osp
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.utils import train_test_split_edges
from torch_geometric.nn import GAE, VGAE, APPNP
import torch_geometric.transforms as T
from loader import Loader

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='VGNAE')
parser.add_argument('--dataset', type=str, default='celegans')
parser.add_argument('--epochs', type=int, default=300)
parser.add_argument('--channels', type=int, default=128)
parser.add_argument('--scaling_factor', type=float, default=1.8)
parser.add_argument('--training_rate', type=float, default=0.8) 
args = parser.parse_args()
dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

loader = Loader(args.dataset)
data = loader.process()
data = T.NormalizeFeatures()(data)

class Encoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels, edge_index = None):
        super(Encoder, self).__init__()
        self.linear1 = nn.Linear(in_channels, out_channels)
        self.linear2 = nn.Linear(in_channels, out_channels)
        self.propagate = APPNP(K=1, alpha=0)

    def forward(self, x, edge_index,not_prop=0):
        if args.model == 'GNAE':
            x = self.linear1(x)
            x = F.normalize(x,p=2,dim=1)  * args.scaling_factor
            x = self.propagate(x, edge_index)
            return x

        if args.model == 'VGNAE':
            x_ = self.linear1(x)
            x_ = self.propagate(x_, edge_index)

            x = self.linear2(x)
            x = F.normalize(x,p=2,dim=1) * args.scaling_factor
            x = self.propagate(x, edge_index)
            return x, x_

        return x


channels = args.channels
train_rate = args.training_rate
val_ratio = (1-args.training_rate) / 3
test_ratio = (1-args.training_rate) / 3 * 2

data = train_test_split_edges(data, val_ratio=val_ratio, test_ratio=test_ratio)
data.train_mask = data.val_mask = data.test_mask = data.y = None
x, train_pos_edge_index = data.x, data.train_pos_edge_index

if args.model == 'GNAE':   
    model = GAE(Encoder(data.x.size()[1], channels, data.train_pos_edge_index.to(dev))).to(dev)
if args.model == 'VGNAE':
    model = VGAE(Encoder(data.x.size()[1], channels)).to(dev)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    

def train(epoch:int, x, train_pos_edge_index):
    model.train()
    optimizer.zero_grad()
    z  = model.encode(x.to(dev), train_pos_edge_index.to(dev))

    if epoch == args.epochs-1:
        features_vector = z.cpu().detach().numpy()
        np.save(f"data/dti_vectors/{args.dataset}_{args.channels}",features_vector)

    loss = model.recon_loss(z, train_pos_edge_index)
    if args.model in ['VGAE']:
        loss = loss + (1 / data.num_nodes) * model.kl_loss()
    loss.backward()
    optimizer.step()
    return loss

def test_binary(x, pos_edge_index, neg_edge_index):
    model.eval()
    with torch.no_grad():
        z = model.encode(x.to(dev), train_pos_edge_index.to(dev))
    return model.test(z, pos_edge_index, neg_edge_index)

for epoch in range(1,args.epochs):
    loss = train(epoch, x, train_pos_edge_index)
    loss = float(loss)
    
    with torch.no_grad():
        auc, ap = test_binary(x, data.test_pos_edge_index, data.test_neg_edge_index)
        print('Epoch: {:03d}, LOSS: {:.4f}, AUC: {:.4f}, AP: {:.4f}'.format(epoch, loss, auc, ap))
