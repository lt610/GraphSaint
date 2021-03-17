import torch
import dgl

g = dgl.graph((torch.tensor([0, 0, 1, 1]), torch.tensor([1, 0, 2, 3])))
print(g.in_edges(torch.tensor([1, 0]), form='eid'))