import json
from functools import namedtuple
import torch as th
import dgl
import numpy as np
import scipy.sparse as sp
import scipy
from dgl.data import CoraGraphDataset, PPIDataset
from sklearn.preprocessing import StandardScaler
import random


a = th.Tensor([1, 2, 3, 4])
b = th.LongTensor([1, 2])
a[b] += 1
print(a)




