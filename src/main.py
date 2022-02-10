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
#np.set_printoptions(threshold=np.inf)

seed = 234
np.random.seed(seed)
tf.set_random_seed(seed)
parser = argparse.ArgumentParser()


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
# pubmed
parser.add_argument('--dataset', type=str, default='pubmed', help='which dataset to use')
parser.add_argument('--epochs', type=int, default=200, help='the number of epochs')
parser.add_argument('--dim', type=int, default=32, help='dimension of hidden layers')
parser.add_argument('--gcn_layer', type=int, default=2, help='number of GCN layers')
parser.add_argument('--lpa_iter', type=int, default=1, help='number of LPA iterations')
parser.add_argument('--l2_weight', type=float, default=2e-4, help='weight of l2 regularization')
parser.add_argument('--lpa_weight', type=float, default=1, help='weight of LP regularization')
parser.add_argument('--dropout', type=float, default=0, help='dropout rate')
parser.add_argument('--lr', type=float, default=0.1, help='learning rate')
'''

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

print("LPA_adj",LPA_adj)
print("LPA_adj[0]",LPA_adj[0])
print("Type LPA_adj[0]",type(LPA_adj[0]))
print("LPA_adj[1]",LPA_adj[1])
print("Type LPA_adj[1]",type(LPA_adj[1]))
print("LPA_adj[0].T[0])",LPA_adj[0].T[0])
print("LPA_adj[0].T[1])",LPA_adj[0].T[1])



#----------------"method1"---------------------#
# sp_edges,sp_weights = reduction('edges','delete','all',1.0/8,1.0/4,10240,False,False,LPA_adj[0],LPA_adj[1])
# #10240:target number of edges
# print("sp_edges",sp_edges)
# print("sp_weights",sp_weights)
# SP_adj = tf.SparseTensor(indices = sp_edges,values = sp_weights,dense_shape=[2708, 2708])
# print(SP_adj)
# print(type(SP_adj))
#----------------"method1"---------------------#


#----------------"method2"---------------------#
##LPA_adj[1]:edge weights; LPA_adj[0]：edges; LPA_adj[0].T[0] and LPA_adj[0].T[1]: endpoint sets of edges

normalized_weight =np.array(LPA_adj[1])
normalized_weight=normalized_weight-np.min(normalized_weight)
normalized_weight=normalized_weight/np.max(normalized_weight)

A = sp.coo_matrix((np.abs(normalized_weight),(list(LPA_adj[0].T[0]), list(LPA_adj[0].T[1]))),shape=[2708, 2708])  #2708:size of nodes
A = A.tocsr()
A = A + A.T
print("A",A)
B = spectral_sparsify(A, q = None,epsilon=2e-2, log_every=100, convergence_after=100, eta=1.2e-5, max_iters=100000, prevent_vertex_blow_up=True)
print(f'Sparsified graph has {B.nnz} edges.')
rows, cols = B.nonzero()
weights = np.array(B[rows, cols].tolist())
sp_edges = np.array([rows,cols]).T
#print(weights)
print("Sp_weights",weights[0])
print("Type_Sp_weights",type(weights))
print("sp_edges",np.array([rows,cols]).T)
SP_adj = tf.SparseTensor(indices = sp_edges,values = weights[0],dense_shape=[2708, 2708])
print(SP_adj)
print(type(SP_adj))
# #----------------"method2"---------------------#

train_GCN(args, features, labels, SP_adj, train_mask, val_mask, test_mask)
