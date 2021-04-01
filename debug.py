import json
import math
import torch as th
import dgl
import numpy as np
import scipy.sparse as sp
import scipy
from dgl.data import CoraGraphDataset, PPIDataset
from sklearn.preprocessing import StandardScaler
import random


class Node(object):
    def __init__(self, x):
        self.x = x
        self.left = None
        self.right = None


def mid_search(root, result):
    if root is None:
        return
    result.append(root.x)
    mid_search(root.left)
    mid_search(root.right)


def solution(root):
    seq = []
    mid_search(root, seq)
    result = []
    pre = None
    for i in range(len(seq)):
        if pre is None:
            pre = seq[i]
        if seq[i] > pre:
            result.append(seq[i])
        else:
            pre = seq[i]
    return result


a = th.Tensor([1, 2, 3 , 4])
print(a.square())

