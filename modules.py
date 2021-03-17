from dgl.nn.pytorch.conv import GraphConv
import torch.nn as nn
import torch.nn.functional as F


class GCNNet(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, n_layers=2, activation=F.relu):
        super(GCNNet, self).__init__()
        self.gcn = nn.ModuleList()
        self.gcn.append(GraphConv(in_feats=in_dim, out_feats=hid_dim, activation=activation))
        for _ in range(n_layers - 2):
            self.gcn.append(GraphConv(in_feats=hid_dim, out_feats=hid_dim, activation=activation))
        self.gcn.append(GraphConv(in_feats=hid_dim, out_feats=out_dim, activation=None))

    def forward(self, graph, features):
        h = features
        for layer in self.gcn:
            h = layer(graph, h)
        return h
