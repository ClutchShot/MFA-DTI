import argparse
import glob
import os
import time

import torch
import torch.nn.functional as F
from models import Model
from torch.utils.data import random_split
from torch_geometric.data import DataLoader
from torch_geometric.datasets import TUDataset
from utils import read, read_amino, graph_construction_and_featurization
import numpy as np

parser = argparse.ArgumentParser()

parser.add_argument('--seed', type=int, default=777, help='random seed')
parser.add_argument('--batch_size', type=int, default=512, help='batch size')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=0.001, help='weight decay')
parser.add_argument('--nhid', type=int, default=128, help='hidden size')
parser.add_argument('--sample_neighbor', type=bool, default=True, help='whether sample neighbors')
parser.add_argument('--sparse_attention', type=bool, default=True, help='whether use sparse attention')
parser.add_argument('--structure_learning', type=bool, default=True, help='whether perform structure learning')
parser.add_argument('--pooling_ratio', type=float, default=0.5, help='pooling ratio')
parser.add_argument('--dropout_ratio', type=float, default=0.0, help='dropout ratio')
parser.add_argument('--lamb', type=float, default=1.0, help='trade-off parameter')
parser.add_argument('--dataset', type=str, default='drugbank')
parser.add_argument('--device', type=str, default='cuda:0', help='specify cuda devices')
parser.add_argument('--epochs', type=int, default=100, help='maximum number of epochs')
parser.add_argument('--patience', type=int, default=100, help='patience for early stopping')

args = parser.parse_args()
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)


if args.dataset == 'amino_acids':
    amino_names, raw_data = read_amino(args.dataset)
    dataset  = graph_construction_and_featurization(raw_data)
    raw_data = amino_names
    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    val_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    args.num_classes = 2
    args.num_features = 1
else:
    raw_data = read(args.dataset)
    print(f"SIZE:{len(raw_data)}")
    dataset  = graph_construction_and_featurization(raw_data)
    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    val_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    args.num_classes = 2
    args.num_features = 1

model = Model(args).to(args.device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)


def train():
    min_loss = 1e10
    patience_cnt = 0
    val_loss_values = []
    best_epoch = 0

    t = time.time()
    model.train()
    for epoch in range(args.epochs):
        loss_train = 0.0
        correct = 0
        for i, data in enumerate(train_loader):
            optimizer.zero_grad()
            data = data.to(args.device)
            out, out_vector = model(data)
            loss = F.nll_loss(out, data.y)
            loss.backward()
            optimizer.step()
            loss_train += loss.item()
            pred = out.max(dim=1)[1]
            correct += pred.eq(data.y).sum().item()

        acc_train = correct / len(train_loader.dataset)
        print('Epoch: {:04d}'.format(epoch + 1), 'loss_train: {:.6f}'.format(loss_train),
              'acc_train: {:.6f}'.format(acc_train), 'time: {:.6f}s'.format(time.time() - t))

        torch.save(model.state_dict(), '{}.pth'.format(epoch))
        if val_loss_values[-1] < min_loss:
            min_loss = val_loss_values[-1]
            best_epoch = epoch
            patience_cnt = 0
        else:
            patience_cnt += 1

        if patience_cnt == args.patience:
            break

        files = glob.glob('*.pth')
        for f in files:
            epoch_nb = int(f.split('.')[0])
            if epoch_nb < best_epoch:
                os.remove(f)

    files = glob.glob('*.pth')
    for f in files:
        epoch_nb = int(f.split('.')[0])
        if epoch_nb > best_epoch:
            os.remove(f)
    print('Optimization Finished! Total time elapsed: {:.6f}'.format(time.time() - t))

    return best_epoch


def safe_vectors(loader):
    model.eval()
    correct = 0.0
    loss_test = 0.0
    features_vector = []
    
    for data in loader:
        data = data.to(args.device)
        out, out_vector = model(data)
        features_vector.extend(out_vector.cpu().detach().numpy())

    features_vector_dict = {raw_data[i]:features_vector[i] for i in range(len(raw_data))}
    np.save(f"data/drug/{args.dataset}_{args.nhid}", features_vector_dict)


if __name__ == '__main__':
    # Pre-training vectors
    best_model = train()
    safe_vectors(test_loader)


