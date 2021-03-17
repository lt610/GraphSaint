import os
import torch as th
import random
import numpy as np
import dgl.function as fn
import torch
import dgl
from dgl.sampling import random_walk


class SAINTSampler(object):
    def __init__(self, dn, g, train_nid, num_batch, num_repeat=50):
        self.train_g: dgl.graph = g.subgraph(train_nid)
        self.dn, self.num_batch, self.num_repeat = dn, num_batch, num_repeat
        self.node_counter = th.zeros((self.train_g.num_nodes(),))
        self.edge_counter = th.zeros((self.train_g.num_edges(),))
        self.prob = None

        graph_fn, norm_fn = self.__generate_fn__()

        if os.path.exists(graph_fn):
            self.subgraphs = np.load(graph_fn, allow_pickle=True)
            self.alp, self.lam = np.load(norm_fn, allow_pickle=True)
        else:
            os.makedirs('./datasets/', exist_ok=True)

            self.subgraphs = []
            self.N = sampled_nodes = 0
            while sampled_nodes < self.train_g.num_nodes() * num_repeat:
                subgraph = self.__sample__()
                self.subgraphs.append(subgraph)
                sampled_nodes += subgraph.shape[0]
                self.N += 1
            np.save(graph_fn, self.subgraphs)

            self.aggr_norm, self.loss_norm = self.__compute_norm__()
            np.save(norm_fn, (self.alp, self.lam))

            self.__clear__()

    def __clear__(self):
        self.prob = None
        self.node_counter = None
        self.edge_counter = None

    def __counter__(self, sampled_nodes):
        self.node_counter[sampled_nodes] += 1
        in_edges, out_edges = self.train_g.in_edges(sampled_nodes, form="eid"),\
                              self.train_g.out_edges(sampled_nodes, form="eid")
        sampled_edges = th.cat(in_edges, out_edges)
        self.edge_counter[sampled_edges] += 1

    def __generate_fn__(self):
        raise NotImplementedError


    def __compute_norm__(self):
        loss_norm = self.N / self.node_counter / self.train_g.num_nodes()



        aggr_norm, loss_norm = 1, 1
        return aggr_norm, loss_norm

    def __sample__(self):
        raise NotImplementedError

    def __len__(self):
        return self.num_batch

    def __iter__(self):
        self.n = 0
        return self

    def __next__(self):
        if self.n < self.num_batch:
            self.n += 1
            return self.train_g.subgraph(self.subgraphs[self.n])
        else:
            random.shuffle(self.subgraphs)
            return StopIteration


class SAINTNodeSampler(SAINTSampler):
    def __init__(self, node_budget, dn, g, train_nid, num_batch, num_repeat=50):
        self.node_budget = node_budget
        super(SAINTNodeSampler, self).__init__(dn, g, train_nid, num_batch, num_repeat)

    def __generate_fn__(self):
        graph_fn = os.path.join('./datasets/{}_Node_{}_{}.npy'.format(self.dn, self.node_budget,
                                                                      self.num_repeat))
        norm_fn = os.path.join('./datasets/{}_Node_{}_{}_norm.npy'.format(self.dn, self.node_budget,
                                                                          self.num_repeat))
        return graph_fn, norm_fn

    def __sample__(self):
        if self.prob is None:
            degrees = self.train_g.in_degrees()
            self.prob = degrees ** 2
        sampled_nodes = th.multinomial(self.prob, num_samples=self.node_budget, replacement=True).unique()
        self.__counter__(sampled_nodes)

        return sampled_nodes.numpy()


class SAINTEdgeSampler(SAINTSampler):
    def __init__(self, edge_budget, dn, g, train_nid, num_batch, num_repeat=50):
        self.edge_budget = edge_budget
        super(SAINTEdgeSampler, self).__init__(dn, g, train_nid, num_batch, num_repeat)

    def __generate_fn__(self):
        graph_fn = os.path.join('./datasets/{}_Edge_{}_{}.npy'.format(self.dn, self.edge_budget,
                                                                      self.num_repeat))
        norm_fn = os.path.join('./datasets/{}_Edge_{}_{}_norm.npy'.format(self.dn, self.edge_budget,
                                                                          self.num_repeat))
        return graph_fn, norm_fn

    def __sample__(self):
        if self.prob is None:
            src, dst = self.train_g.edges()
            src_degrees, dst_degrees = self.train_g.in_degrees(src), self.train_g.in_degrees(dst)
            self.prob = 1. / src_degrees + 1. /dst_degrees
        sampled_edges = th.multinomial(self.prob, num_samples=self.edge_budget, replacement=True).unique()
        sampled_src, sampled_dst = self.train_g.find_edges(sampled_edges)
        sampled_nodes = th.cat([sampled_src, sampled_dst]).unique()
        self.__counter__(sampled_nodes)
        return sampled_nodes.numpy()


class SAINTRandomWalkSampler(SAINTSampler):
    def __init__(self, num_roots, length, dn, g, train_nid, num_batch, num_repeat=50):
        self.num_roots, self.length = num_roots, length
        super(SAINTRandomWalkSampler, self).__init__(dn, g, train_nid, num_batch, num_repeat)

    def __generate_fn__(self):
        graph_fn = os.path.join('./datasets/{}_RW_{}_{}_{}.npy'.format(self.dn, self.num_roots,
                                                                        self.length, self.num_repeat))
        norm_fn = os.path.join('./datasets/{}_RW_{}_{}_{}_norm.npy'.format(self.dn, self.num_roots,
                                                                           self.length, self.num_repeat))
        return graph_fn, norm_fn

    def __sample__(self):
        sampled_roots = th.randint(0, self.train_g.num_nodes(), (self.num_roots, )).unique()
        traces, _ = random_walk(self.train_g, nodes=sampled_roots, length=self.length)
        sampled_nodes = traces.reshape(-1).unique()
        self.__counter__(sampled_nodes)
        return sampled_nodes.numpy()



