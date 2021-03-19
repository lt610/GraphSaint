import torch.nn as nn
import torch.nn.functional as F
import torch as th
import dgl.function as fn


class GCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim, bias=True,
                 activation=None, graph_norm=True):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=bias)
        self.activation = activation
        self.graph_norm = graph_norm
        self.reset_parameters()

    def reset_parameters(self):
        self.linear.reset_parameters()

    def forward(self, graph, features):
        g = graph.local_var()
        if self.graph_norm:
            degs = g.in_degrees().float()
            norm = th.pow(degs, -0.5)
            norm[th.isinf(norm)] = 0
            norm = norm.to(features.device).unsqueeze(1)

        h = features * norm
        g.ndata['h'] = h
        # w is the weights of edges
        if 'w' not in g.edata:
            g.edata['w'] = th.ones((g.num_edges(), )).to(features.device)
        g.update_all(fn.u_mul_e('h', 'w', 'm'),
                     fn.sum('m', 'h'))
        h = g.ndata.pop('h')
        if self.graph_norm:
            h = h * norm
        h = self.linear(h)
        if self.activation is not None:
            h = self.activation(h)
        return h


class GCNNet(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, n_layers=2, activation=F.relu):
        super(GCNNet, self).__init__()
        self.gcn = nn.ModuleList()
        self.gcn.append(GCNLayer(in_dim=in_dim, out_dim=hid_dim, activation=activation))
        for _ in range(n_layers - 2):
            self.gcn.append(GCNLayer(in_dim=hid_dim, out_dim=hid_dim, activation=activation))
        self.gcn.append(GCNLayer(in_dim=hid_dim, out_dim=out_dim, activation=None))

    def forward(self, graph):
        h = graph.ndata['feat']
        for layer in self.gcn:
            h = layer(graph, h)
        return h
