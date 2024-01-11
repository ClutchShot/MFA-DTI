from torch import nn
from Conv import Conv
import torch


class Fusion(nn.Module):
    def __init__(self, args):
        super(Fusion, self).__init__()
        self.device = args.device
        if self.device != 'cpu':
            self.encoder_layer = nn.TransformerEncoderLayer(d_model=328, nhead=8)
        else:
            # torch 1.11
            # cpu
            self.encoder_layer = nn.TransformerEncoderLayer(d_model=328, nhead=8, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=3)
        self.conv = Conv(args) 

        self.layer1 = nn.Sequential(nn.Linear(328,128), nn.BatchNorm1d(128), nn.GELU())
        self.layer2 = nn.Sequential(nn.Linear(128,64), nn.BatchNorm1d(64), nn.GELU())
        self.layer3 = nn.Sequential(nn.Linear(64,1), nn.Sigmoid())
        

    def forward(self, data_drug, data_prot, data_graph):

        X_2 = self.conv(data_prot)
        X = torch.cat((data_drug, X_2, data_graph), dim=1)
        X = X.to(torch.float32)

        if self.device != 'cpu':
            X = X.unsqueeze(dim=0)

        X = self.encoder(X)

        if self.device != 'cpu':
            X = X.squeeze(dim=0)
            
        X = self.layer1(X)
        X = self.layer2(X)
        X = self.layer3(X)

        return X