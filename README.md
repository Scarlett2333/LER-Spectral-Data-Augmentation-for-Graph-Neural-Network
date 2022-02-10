# LER

This repository is the implementation of our project of SJTU-CS222 Algorithm Design & Analysis:
> LER：Spectral Data Augmentation for Graph Neural Network 
> Wenyi Xue, Sijia Li, Luyuan Jin  


### learnable LPA
Use the first part of GCN-LPA, an end-to-end model that unifies Graph Convolutional Neural Networks (GCN) and Label Propagation Algorithm (LPA) for adaptive semi-supervised node classification, proposed by Hongwei Wang and Jure Leskovec in the paper <Unifying Graph Convolutional Neural Networks and Label Propagation>.


### Files in the folder

- `data/`
  - `citeseer/`
  - `cora/`
  - `pubmed/`
  - `ms_academic_cs.npz` (Coauthor-CS)
  - `ms_academic_phy.npz` (Coauthor-Phy)
- `src/`: implementation of LER.


### Sparsification Algorithms of unweighted/weighted Graph
> Graph Sparsification by Effective Resistances, Daniel A. Spielman, Nikhil Srivastava, 2009.
> A Unifying Framework for Spectrum-Preserving Graph Sparsification and Coarsening, Gecia Bravo-Hermsdorff, Lee M. Gunderson, 2020.


### Running the code

```
$ python main.py
```
**Note**: The default dataset is Citeseer.
Hyper-parameter settings for other datasets are provided in ``main.py``.


### Required packages

The code has been tested running under Python 3.6.5, with the following packages installed (along with their dependencies):

- tensorflow == 1.12.0
- networkx == 2.1
- numpy == 1.14.3
- scipy == 1.1.0
- sklearn == 0.19.1
- matplotlib == 2.2.2
