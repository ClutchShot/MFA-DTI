from torch import nn
import torch
from utlis import *
import numpy as np
from Fusion import Fusion
import torch.utils.data as Data
import argparse
from sklearn.metrics import *
import sklearn.metrics as m
import random
from numpy.random import default_rng, shuffle
from tqdm import tqdm

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--device', type=str, default='cpu')
parser.add_argument('--dataset', type=str, default='drugbank')
parser.add_argument('--dti_index', type=bool, default=False)
parser.add_argument('--acivation', type=str, default='gelu')
parser.add_argument('--task', type=str, default='-1')

args = parser.parse_args()
device = args.device
DATASET = args.dataset

drug_file = f'data/drug/{DATASET}_128.npy'
protein_file = f'data/protein/{DATASET}_protein_1024_128_max.npy'
graph_file = f'data/graph/{DATASET}_graph_s.npy'
dti_file = f"data/dti/{DATASET}.txt"
print(protein_file)
print(graph_file)
print(dti_file)

dti_raw = read_dti(dti_file, drugbank=args.dti_index)
print(f"DATASET:{DATASET}  Size:{len(dti_raw)}  Device:{device}")

drug_vectors = np.load(drug_file, allow_pickle=True).item()
protein_vectors = np.load(protein_file, allow_pickle=True).item()
graph_vectors = np.load(graph_file, allow_pickle=True)

random_mask = list(range(len(dti_raw)))
shuffle(random_mask)
dti = [dti_raw[i] for i in random_mask]

 
X_g = np.array([graph_vectors[i] for i in random_mask])
X_d = to_vec(dti,drug_vectors, flag=0)
X_p = to_vec(dti,protein_vectors, flag=1)

Y = np.array([int(i[2]) for i in dti], dtype=float) 

train_size = int(len(dti) * 0.8)

train_xg = torch.from_numpy(X_g[:train_size])
train_xd = torch.from_numpy(X_d[:train_size])
train_xp = torch.from_numpy(X_p[:train_size])
train_y = torch.from_numpy(Y[:train_size])

test_xg = torch.from_numpy(X_g[train_size:])
test_xd = torch.from_numpy(X_d[train_size:])
test_xp = torch.from_numpy(X_p[train_size:])
test_y = torch.from_numpy(Y[train_size:])

torch_dataset_train = Data.TensorDataset(train_xd, train_xp, train_xg, train_y)
torch_dataset_test = Data.TensorDataset(test_xd, test_xp, test_xg, test_y)


loader_train = Data.DataLoader(dataset=torch_dataset_train, batch_size=1024, shuffle=True)
loader_test = Data.DataLoader(dataset=torch_dataset_test, batch_size=512, shuffle=False)

model = Fusion(args).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
loss_func = torch.nn.BCEWithLogitsLoss()


iter_loss = 0
final = 0
best_metrics = {'Auc':0,'Aupr':0,'Rec':0,'Pre':0, 'F1':0}
best_auc = 0
balance_metrics = {'Auc':0,'Aupr':0,'Rec':0,'Pre':0, 'F1':0}

balance = 0
for epoch in range(0, args.epochs):
    # TRAINING
    for step, (batch_xd, batch_xp, batch_xg, batch_y) in enumerate(tqdm(loader_train)):
        batch_xd, batch_xp, batch_xg, batch_y = batch_xd.to(device), batch_xp.to(device), batch_xg.to(device), batch_y.to(device)
        model.train()
        out = model(batch_xd, batch_xp,batch_xg)
        optimizer.zero_grad()
        out = out.view(-1)
        loss = loss_func(out, batch_y)
        loss.backward()
        optimizer.step()
        iter_loss += loss.item()
        final = iter_loss / (step + 1)


    # EVALUATION
    model.eval()
    pred_all = np.array([])
    labels_all = np.array([])
    with torch.no_grad():
        for step, (batch_xd, batch_xp, batch_xg, batch_y) in enumerate(loader_test):
            batch_xd, batch_xp, batch_xg, batch_y = batch_xd.to(device), batch_xp.to(device), batch_xg.to(device), batch_y.to(device)
            out = model(batch_xd, batch_xp, batch_xg)
            out = out.view(-1)
            loss = loss_func(out, batch_y)
            pred = out.cpu().detach().numpy()
            labels = batch_y.cpu().detach().numpy()
            pred_all = np.append(pred_all, pred)
            labels_all = np.append(labels_all, labels)


        
        auc = roc_auc_score(labels_all, pred_all)
        ap = average_precision_score(labels_all, pred_all)
        p, r, t = precision_recall_curve(labels_all, pred_all)
        aupr = m.auc(r, p)
        lab = [0.0 if l < 0.5 else 1.0 for l in labels_all]
        scr = [0.0 if s < 0.5 else 1.0 for s in pred_all]
        scr = np.array(scr, dtype=float)
        precision = precision_score(labels_all, scr)
        recall = recall_score(labels_all, scr)
        f1 = f1_score(labels_all, scr)

        if auc > best_auc:
            best_auc = auc
            best_metrics['Auc'] = auc
            best_metrics['Aupr'] = aupr
            best_metrics['Rec'] = recall
            best_metrics['Pre'] = precision
            best_metrics['F1'] = f1

        if precision + recall > balance:
            balance = precision + recall
            balance_metrics['Auc'] = auc
            balance_metrics['Aupr'] = aupr
            balance_metrics['Rec'] = recall
            balance_metrics['Pre'] = precision
            balance_metrics['F1'] = f1

        print(f"Epoch:{epoch}, AUC:{auc:.4f}, PRE:{precision:.4f}, REC:{recall:.4f}, AUPR:{aupr:.4f}, F1:{f1:.4f}")


print(f"Final score: {best_metrics}")
print(f"Balance score: {balance_metrics}")


