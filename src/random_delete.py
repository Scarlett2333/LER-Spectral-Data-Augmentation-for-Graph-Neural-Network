import argparse
import numpy as np
import tensorflow as tf
from time import time
from data_loader import load_data, load_npz, load_random
from train import train_LPA,train_GCN
from graph_reduction import reduction
from spectral_sparisifcation import spectral_sparsify
import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
import networkx as nx
import random
#np.set_printoptions(threshold=np.inf)

seed = 235
np.random.seed(seed)
tf.set_random_seed(seed)
parser = argparse.ArgumentParser()

'''
# cora
parser.add_argument('--dataset', type=str, default='cora', help='which dataset to use')
parser.add_argument('--epochs', type=int, default=200, help='the number of epochs')
parser.add_argument('--dim', type=int, default=32, help='dimension of hidden layers')
parser.add_argument('--gcn_layer', type=int, default=5, help='number of GCN layers')
parser.add_argument('--lpa_iter', type=int, default=5, help='number of LPA iterations')
parser.add_argument('--l2_weight', type=float, default=1e-4, help='weight of l2 regularization')
parser.add_argument('--lpa_weight', type=float, default=10, help='weight of LP regularization')
parser.add_argument('--dropout', type=float, default=0.2, help='dropout rate')
parser.add_argument('--lr', type=float, default=0.05, help='learning rate')
'''

'''
# citeseer
parser.add_argument('--dataset', type=str, default='citeseer', help='which dataset to use')
parser.add_argument('--epochs', type=int, default=200, help='the number of epochs')
parser.add_argument('--dim', type=int, default=16, help='dimension of hidden layers')
parser.add_argument('--gcn_layer', type=int, default=2, help='number of GCN layers')
parser.add_argument('--lpa_iter', type=int, default=5, help='number of LPA iterations')
parser.add_argument('--l2_weight', type=float, default=5e-4, help='weight of l2 regularization')
parser.add_argument('--lpa_weight', type=float, default=1, help='weight of LP regularization')
parser.add_argument('--dropout', type=float, default=0, help='dropout rate')
parser.add_argument('--lr', type=float, default=0.2, help='learning rate')
'''

# pubmed
parser.add_argument('--dataset', type=str, default='pubmed', help='which dataset to use')
parser.add_argument('--epochs', type=int, default=200, help='the number of epochs')
parser.add_argument('--dim', type=int, default=32, help='dimension of hidden layers')
parser.add_argument('--gcn_layer', type=int, default=2, help='number of GCN layers')
parser.add_argument('--lpa_iter', type=int, default=20, help='number of LPA iterations')
parser.add_argument('--l2_weight', type=float, default=2e-4, help='weight of l2 regularization')
parser.add_argument('--lpa_weight', type=float, default=1, help='weight of LP regularization')
parser.add_argument('--dropout', type=float, default=0, help='dropout rate')
parser.add_argument('--lr', type=float, default=0.1, help='learning rate')


'''
# coauthor-cs
parser.add_argument('--dataset', type=str, default='coauthor-cs', help='which dataset to use')
parser.add_argument('--epochs', type=int, default=200, help='the number of epochs')
parser.add_argument('--dim', type=int, default=32, help='dimension of hidden layers')
parser.add_argument('--gcn_layer', type=int, default=2, help='number of GCN layers')
parser.add_argument('--lpa_iter', type=int, default=2, help='number of LPA iterations')
parser.add_argument('--l2_weight', type=float, default=1e-4, help='weight of l2 regularization')
parser.add_argument('--lpa_weight', type=float, default=2, help='weight of LP regularization')
parser.add_argument('--dropout', type=float, default=0.2, help='dropout rate')
parser.add_argument('--lr', type=float, default=0.1, help='learning rate')
'''

'''
# coauthor-phy
parser.add_argument('--dataset', type=str, default='coauthor-phy', help='which dataset to use')
parser.add_argument('--epochs', type=int, default=200, help='the number of epochs')
parser.add_argument('--dim', type=int, default=32, help='dimension of hidden layers')
parser.add_argument('--gcn_layer', type=int, default=2, help='number of GCN layers')
parser.add_argument('--lpa_iter', type=int, default=3, help='number of LPA iterations')
parser.add_argument('--l2_weight', type=float, default=1e-4, help='weight of l2 regularization')
parser.add_argument('--lpa_weight', type=float, default=1, help='weight of LP regularization')
parser.add_argument('--dropout', type=float, default=0.2, help='dropout rate')
parser.add_argument('--lr', type=float, default=0.05, help='learning rate')
'''

'''
# random graph
# this is only for calculating the training time
parser.add_argument('--dataset', type=str, default='random', help='which dataset to use')
parser.add_argument('--epochs', type=int, default=100, help='the number of epochs')
parser.add_argument('--dim', type=int, default=16, help='dimension of hidden layers')
parser.add_argument('--gcn_layer', type=int, default=5, help='number of GCN layers')
parser.add_argument('--lpa_iter', type=int, default=6, help='number of LPA iterations')
parser.add_argument('--l2_weight', type=float, default=5e-8, help='weight of l2 regularization')
parser.add_argument('--lpa_weight', type=float, default=15, help='weight of LP regularization')
parser.add_argument('--dropout', type=float, default=0.2, help='dropout rate')
parser.add_argument('--lr', type=float, default=0.1, help='learning rate')
'''

t = time()
args = parser.parse_args()

if args.dataset in ['cora', 'citeseer', 'pubmed']:
    #data = list(load_data(args.dataset))
    data = list(load_data(args.dataset))
    print(type(data))

elif args.dataset in ['coauthor-cs', 'coauthor-phy']:
    data = list(load_npz(args.dataset))
else:
    n_nodes = 1000
    data = list(load_random(n_nodes=n_nodes, n_train=100, n_val=200, p=10/n_nodes))

features, labels, adj, train_mask, val_mask, test_mask = [data[i] for i in range(6)]
LPA_adj = train_LPA(args, data)

edge_pair=np.array(LPA_adj[0])
normalized_weight =np.array(LPA_adj[1])
normalized_weight=normalized_weight-np.min(normalized_weight)
normalized_weight=normalized_weight/np.max(normalized_weight)

for i in range(len(adj[0])//7):
    a=random.randint(0,labels.shape[1]-1)
    np.delete(edge_pair,a,0)
    np.delete(normalized_weight,a,None) 
print(edge_pair,normalized_weight)
SP_adj = tf.SparseTensor(indices = edge_pair.astype(np.float64),values = normalized_weight.astype(np.float64),dense_shape=[features[2][0], features[2][0]])


train_GCN(args, features, labels, SP_adj, train_mask, val_mask, test_mask)
