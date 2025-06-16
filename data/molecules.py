import torch
import pickle
import torch.utils.data
import time
import os
import numpy as np

import csv

import dgl
import pandas as pd
import random
from scipy import sparse as sp
import numpy as np
import torch.nn.functional as F
from dgllife.utils import BaseAtomFeaturizer,WeaveAtomFeaturizer, CanonicalBondFeaturizer, smiles_to_bigraph, EarlyStopping
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import train_test_split

from functools import partial

from deepchem.splits import ScaffoldSplitter
from deepchem.data import Dataset
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from collections import defaultdict
from sklearn.utils import shuffle

# *NOTE
# The dataset pickle and index files are in ./zinc_molecules/ dir
# [<split>.pickle and <split>.index; for split 'train', 'val' and 'test']

class MoleculeQM8DGL(torch.utils.data.Dataset):
    def __init__(self, data_dir, split, graphs, labels, num_graphs=None):
        self.data_dir = data_dir
        self.split = split
        self.num_graphs = num_graphs

        df = pd.read_csv('%s.csv' % (data_dir))

        # 直接获取 'measured log solubility in mols per litre' 列作为标签，用于回归任务
        self.labels = df.iloc[:, 1:13].values

        # 整体数据集索引
        all_indices = np.arange(len(df))

        # 划分训练集、验证集和测试集的比例，这里可以根据实际需求调整
        test_size = 0.1  # 测试集占比20%
        val_size = 0.1  # 验证集占比10%

        # 首先划分出训练集和临时集（验证集 + 测试集）
        train_indices, temp_indices = train_test_split(all_indices, test_size=(val_size + test_size), random_state=42)

        # 再从临时集中划分出验证集和测试集
        val_indices, test_indices = train_test_split(temp_indices, test_size=(test_size / (val_size + test_size)), random_state=42)

        graphs = np.array(graphs)

        if split == 'train':
            self.graph_lists = graphs[train_indices].tolist()
            self.graph_labels = self.labels[train_indices].tolist()
            self.n_samples = len(train_indices)
        elif split == 'val':
            self.graph_lists = graphs[val_indices].tolist()
            self.graph_labels = self.labels[val_indices].tolist()
            self.n_samples = len(val_indices)
        elif split == 'test':
            self.graph_lists = graphs[test_indices].tolist()
            self.graph_labels = self.labels[test_indices].tolist()
            self.n_samples = len(test_indices)
        else:
            print(f"Error processing {split}:")

        
    def _prepare(self):
        print("preparing %d graphs for the %s set..." % (self.num_graphs, self.split.upper()))
        
        for molecule in self.graphs:
            node_features = molecule['atom_type'].long()
            
            adj = molecule['bond_type']
            edge_list = (adj != 0).nonzero()  # converting adj matrix to edge_list
            
            edge_idxs_in_adj = edge_list.split(1, dim=1)
            edge_features = adj[edge_idxs_in_adj].reshape(-1).long()
            
            # Create the DGL Graph
            g = dgl.DGLGraph()
            g.add_nodes(molecule['num_atom'])
            g.ndata['feat'] = node_features
            
            for src, dst in edge_list:
                g.add_edges(src.item(), dst.item())
            g.edata['feat'] = edge_features
            
            self.graph_lists.append(g)
            self.graph_labels.append(molecule['logP_SA_cycle_normalized'])
        
    def __len__(self):
        """Return the number of graphs in the dataset."""
        return self.n_samples

    def __getitem__(self, idx):
        """
            Get the idx^th sample.
            Parameters
            ---------
            idx : int
                The sample index.
            Returns
            -------
            (dgl.DGLGraph, int)
                DGLGraph with node feature stored in `feat` field
                And its label.
        """
        return self.graph_lists[idx], self.graph_labels[idx]


class MoleculeESOLDGL(torch.utils.data.Dataset):
    def __init__(self, data_dir, split, graphs, labels, num_graphs=None):
        self.data_dir = data_dir
        self.split = split
        self.num_graphs = num_graphs

        df = pd.read_csv('%s.csv' % (data_dir))

        # 直接获取 'measured log solubility in mols per litre' 列作为标签，用于回归任务
        self.labels = df['measured log solubility in mols per litre'].values

        # 整体数据集索引
        all_indices = np.arange(len(df))

        # 划分训练集、验证集和测试集的比例，这里可以根据实际需求调整
        test_size = 0.1  # 测试集占比20%
        val_size = 0.1  # 验证集占比10%

        # 首先划分出训练集和临时集（验证集 + 测试集）
        train_indices, temp_indices = train_test_split(all_indices, test_size=(val_size + test_size), random_state=42)

        # 再从临时集中划分出验证集和测试集
        val_indices, test_indices = train_test_split(temp_indices, test_size=(test_size / (val_size + test_size)), random_state=42)

        graphs = np.array(graphs)

        if split == 'train':
            self.graph_lists = graphs[train_indices].tolist()
            self.graph_labels = self.labels[train_indices].tolist()
            self.n_samples = len(train_indices)
        elif split == 'val':
            self.graph_lists = graphs[val_indices].tolist()
            self.graph_labels = self.labels[val_indices].tolist()
            self.n_samples = len(val_indices)
        elif split == 'test':
            self.graph_lists = graphs[test_indices].tolist()
            self.graph_labels = self.labels[test_indices].tolist()
            self.n_samples = len(test_indices)
        else:
            print(f"Error processing {split}:")

        
    def _prepare(self):
        print("preparing %d graphs for the %s set..." % (self.num_graphs, self.split.upper()))
        
        for molecule in self.graphs:
            node_features = molecule['atom_type'].long()
            
            adj = molecule['bond_type']
            edge_list = (adj != 0).nonzero()  # converting adj matrix to edge_list
            
            edge_idxs_in_adj = edge_list.split(1, dim=1)
            edge_features = adj[edge_idxs_in_adj].reshape(-1).long()
            
            # Create the DGL Graph
            g = dgl.DGLGraph()
            g.add_nodes(molecule['num_atom'])
            g.ndata['feat'] = node_features
            
            for src, dst in edge_list:
                g.add_edges(src.item(), dst.item())
            g.edata['feat'] = edge_features
            
            self.graph_lists.append(g)
            self.graph_labels.append(molecule['logP_SA_cycle_normalized'])
        
    def __len__(self):
        """Return the number of graphs in the dataset."""
        return self.n_samples

    def __getitem__(self, idx):
        """
            Get the idx^th sample.
            Parameters
            ---------
            idx : int
                The sample index.
            Returns
            -------
            (dgl.DGLGraph, int)
                DGLGraph with node feature stored in `feat` field
                And its label.
        """
        return self.graph_lists[idx], self.graph_labels[idx]


class MoleculeLipophilicityDGL(torch.utils.data.Dataset):
    def __init__(self, data_dir, split, graphs, labels, num_graphs=None):
        self.data_dir = data_dir
        self.split = split
        self.num_graphs = num_graphs

        df = pd.read_csv('%s.csv' % (data_dir))

        # 直接获取 'exp' 列作为标签，用于回归任务
        self.labels = df['exp'].values

        # 整体数据集索引
        all_indices = np.arange(len(df))

        # 划分训练集、验证集和测试集的比例，这里可以根据实际需求调整
        test_size = 0.1  # 测试集占比20%
        val_size = 0.1  # 验证集占比10%

        # 首先划分出训练集和临时集（验证集 + 测试集）
        train_indices, temp_indices = train_test_split(all_indices, test_size=(val_size + test_size), random_state=42)

        # 再从临时集中划分出验证集和测试集
        val_indices, test_indices = train_test_split(temp_indices, test_size=(test_size / (val_size + test_size)), random_state=42)

        graphs = np.array(graphs)

        if split == 'train':
            self.graph_lists = graphs[train_indices].tolist()
            self.graph_labels = self.labels[train_indices].tolist()
            self.n_samples = len(train_indices)
        elif split == 'val':
            self.graph_lists = graphs[val_indices].tolist()
            self.graph_labels = self.labels[val_indices].tolist()
            self.n_samples = len(val_indices)
        elif split == 'test':
            self.graph_lists = graphs[test_indices].tolist()
            self.graph_labels = self.labels[test_indices].tolist()
            self.n_samples = len(test_indices)
        else:
            print(f"Error processing {split}:")

        
    def _prepare(self):
        print("preparing %d graphs for the %s set..." % (self.num_graphs, self.split.upper()))
        
        for molecule in self.graphs:
            node_features = molecule['atom_type'].long()
            
            adj = molecule['bond_type']
            edge_list = (adj != 0).nonzero()  # converting adj matrix to edge_list
            
            edge_idxs_in_adj = edge_list.split(1, dim=1)
            edge_features = adj[edge_idxs_in_adj].reshape(-1).long()
            
            # Create the DGL Graph
            g = dgl.DGLGraph()
            g.add_nodes(molecule['num_atom'])
            g.ndata['feat'] = node_features
            
            for src, dst in edge_list:
                g.add_edges(src.item(), dst.item())
            g.edata['feat'] = edge_features
            
            self.graph_lists.append(g)
            self.graph_labels.append(molecule['logP_SA_cycle_normalized'])
        
    def __len__(self):
        """Return the number of graphs in the dataset."""
        return self.n_samples

    def __getitem__(self, idx):
        """
            Get the idx^th sample.
            Parameters
            ---------
            idx : int
                The sample index.
            Returns
            -------
            (dgl.DGLGraph, int)
                DGLGraph with node feature stored in `feat` field
                And its label.
        """
        return self.graph_lists[idx], self.graph_labels[idx]

class MoleculeFreeSolvDGL(torch.utils.data.Dataset):
    def __init__(self, data_dir, split, graphs, labels, num_graphs=None):
        self.data_dir = data_dir
        self.split = split
        self.num_graphs = num_graphs

        df = pd.read_csv('%s.csv' % (data_dir))

        # 直接获取 'expt' 列作为标签，用于回归任务
        self.labels = df['expt'].values

        # 整体数据集索引
        all_indices = np.arange(len(df))

        # 划分训练集、验证集和测试集的比例，这里可以根据实际需求调整
        test_size = 0.1  # 测试集占比20%
        val_size = 0.1  # 验证集占比10%

        # 首先划分出训练集和临时集（验证集 + 测试集）
        train_indices, temp_indices = train_test_split(all_indices, test_size=(val_size + test_size), random_state=42)

        # 再从临时集中划分出验证集和测试集
        val_indices, test_indices = train_test_split(temp_indices, test_size=(test_size / (val_size + test_size)), random_state=42)

        graphs = np.array(graphs)

        if split == 'train':
            self.graph_lists = graphs[train_indices].tolist()
            self.graph_labels = self.labels[train_indices].tolist()
            self.n_samples = len(train_indices)
        elif split == 'val':
            self.graph_lists = graphs[val_indices].tolist()
            self.graph_labels = self.labels[val_indices].tolist()
            self.n_samples = len(val_indices)
        elif split == 'test':
            self.graph_lists = graphs[test_indices].tolist()
            self.graph_labels = self.labels[test_indices].tolist()
            self.n_samples = len(test_indices)
        else:
            print(f"Error processing {split}:")

        
    def _prepare(self):
        print("preparing %d graphs for the %s set..." % (self.num_graphs, self.split.upper()))
        
        for molecule in self.graphs:
            node_features = molecule['atom_type'].long()
            
            adj = molecule['bond_type']
            edge_list = (adj != 0).nonzero()  # converting adj matrix to edge_list
            
            edge_idxs_in_adj = edge_list.split(1, dim=1)
            edge_features = adj[edge_idxs_in_adj].reshape(-1).long()
            
            # Create the DGL Graph
            g = dgl.DGLGraph()
            g.add_nodes(molecule['num_atom'])
            g.ndata['feat'] = node_features
            
            for src, dst in edge_list:
                g.add_edges(src.item(), dst.item())
            g.edata['feat'] = edge_features
            
            self.graph_lists.append(g)
            self.graph_labels.append(molecule['logP_SA_cycle_normalized'])
        
    def __len__(self):
        """Return the number of graphs in the dataset."""
        return self.n_samples

    def __getitem__(self, idx):
        """
            Get the idx^th sample.
            Parameters
            ---------
            idx : int
                The sample index.
            Returns
            -------
            (dgl.DGLGraph, int)
                DGLGraph with node feature stored in `feat` field
                And its label.
        """
        return self.graph_lists[idx], self.graph_labels[idx]

class MoleculeHIVDGL(torch.utils.data.Dataset):
    def __init__(self, data_dir, split, graphs, labels, num_graphs=None):
        self.data_dir = data_dir
        self.split = split
        self.num_graphs = num_graphs

        df = pd.read_csv('%s.csv' % (data_dir))

        labels = df['HIV_active']

        # 分层抽样分割数据集
        split_num = int(len(df) * 0.1)
        sss = StratifiedShuffleSplit(n_splits=1, test_size=split_num * 4, train_size=split_num * 6, random_state=42)

        # 获取训练集和临时集（验证集+测试集）的索引
        for train_index, temp_index in sss.split(df, labels):
            train_indices = train_index
            temp_indices = temp_index
            temp_labels = labels.iloc[temp_index]

        # 进一步分割临时集为验证集和测试集
        sss_val_test = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
        for val_index, test_index in sss_val_test.split(df.iloc[temp_indices], temp_labels):
            val_indices = temp_indices[val_index]
            test_indices = temp_indices[test_index]

        train_indices = np.array(train_indices)
        val_indices = np.array(val_indices)
        test_indices = np.array(test_indices)

        graphs = np.array(graphs)
        labels = np.array(labels)
        if split == 'train':
            self.graph_lists = graphs[train_indices].tolist()
            self.graph_labels = labels[train_indices].tolist()
            self.n_samples = len(train_indices)
        elif split == 'val':
            self.graph_lists = graphs[val_indices].tolist()
            self.graph_labels = labels[val_indices].tolist()
            self.n_samples = len(val_indices)
        elif split == 'test':
            self.graph_lists = graphs[test_indices].tolist()
            self.graph_labels = labels[test_indices].tolist()
            self.n_samples = len(test_indices)
        else:
            print(f"Error processing {split}:")

                    
        #self._prepare()

        
    def _prepare(self):
        print("preparing %d graphs for the %s set..." % (self.num_graphs, self.split.upper()))
        
        for molecule in self.graphs:
            node_features = molecule['atom_type'].long()
            
            adj = molecule['bond_type']
            edge_list = (adj != 0).nonzero()  # converting adj matrix to edge_list
            
            edge_idxs_in_adj = edge_list.split(1, dim=1)
            edge_features = adj[edge_idxs_in_adj].reshape(-1).long()
            
            # Create the DGL Graph
            g = dgl.DGLGraph()
            g.add_nodes(molecule['num_atom'])
            g.ndata['feat'] = node_features
            
            for src, dst in edge_list:
                g.add_edges(src.item(), dst.item())
            g.edata['feat'] = edge_features
            
            self.graph_lists.append(g)
            self.graph_labels.append(molecule['logP_SA_cycle_normalized'])
        
    def __len__(self):
        """Return the number of graphs in the dataset."""
        return self.n_samples

    def __getitem__(self, idx):
        """
            Get the idx^th sample.
            Parameters
            ---------
            idx : int
                The sample index.
            Returns
            -------
            (dgl.DGLGraph, int)
                DGLGraph with node feature stored in `feat` field
                And its label.
        """
        return self.graph_lists[idx], self.graph_labels[idx]

class MoleculeClinToxDGL(torch.utils.data.Dataset):
    def __init__(self, data_dir, split, graphs, labels, num_graphs=None):
        self.data_dir = data_dir
        self.split = split
        self.num_graphs = num_graphs

        df = pd.read_csv('%s.csv' % (data_dir))

        #labels = df['FDA_APPROVED']
        #labels = df['CT_TOX']

        labels = df.iloc[:, 1:3].values
        stratify_labels = df['FDA_APPROVED'] 

        # 分层抽样分割数据集
        split_num = int(len(df) * 0.1)
        sss = StratifiedShuffleSplit(n_splits=1, test_size=split_num * 2, train_size=split_num * 8, random_state=42)

        # 获取训练集和临时集（验证集+测试集）的索引
        for train_index, temp_index in sss.split(df, stratify_labels):
            train_indices = train_index
            temp_indices = temp_index
            temp_labels = stratify_labels.iloc[temp_index]

        # 进一步分割临时集为验证集和测试集
        sss_val_test = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
        for val_index, test_index in sss_val_test.split(df.iloc[temp_indices], temp_labels):
            val_indices = temp_indices[val_index]
            test_indices = temp_indices[test_index]

        train_indices = np.array(train_indices)
        val_indices = np.array(val_indices)
        test_indices = np.array(test_indices)

        graphs = np.array(graphs)
        labels = np.array(labels)
        if split == 'train':
            self.graph_lists = graphs[train_indices].tolist()
            self.graph_labels = labels[train_indices].tolist()
            self.n_samples = len(train_indices)
        elif split == 'val':
            self.graph_lists = graphs[val_indices].tolist()
            self.graph_labels = labels[val_indices].tolist()
            self.n_samples = len(val_indices)
        elif split == 'test':
            self.graph_lists = graphs[test_indices].tolist()
            self.graph_labels = labels[test_indices].tolist()
            self.n_samples = len(test_indices)
        else:
            print(f"Error processing {split}:")

                    
        #self._prepare()

        
    def _prepare(self):
        print("preparing %d graphs for the %s set..." % (self.num_graphs, self.split.upper()))
        
        for molecule in self.graphs:
            node_features = molecule['atom_type'].long()
            
            adj = molecule['bond_type']
            edge_list = (adj != 0).nonzero()  # converting adj matrix to edge_list
            
            edge_idxs_in_adj = edge_list.split(1, dim=1)
            edge_features = adj[edge_idxs_in_adj].reshape(-1).long()
            
            # Create the DGL Graph
            g = dgl.DGLGraph()
            g.add_nodes(molecule['num_atom'])
            g.ndata['feat'] = node_features
            
            for src, dst in edge_list:
                g.add_edges(src.item(), dst.item())
            g.edata['feat'] = edge_features
            
            self.graph_lists.append(g)
            self.graph_labels.append(molecule['logP_SA_cycle_normalized'])
        
    def __len__(self):
        """Return the number of graphs in the dataset."""
        return self.n_samples

    def __getitem__(self, idx):
        """
            Get the idx^th sample.
            Parameters
            ---------
            idx : int
                The sample index.
            Returns
            -------
            (dgl.DGLGraph, int)
                DGLGraph with node feature stored in `feat` field
                And its label.
        """
        return self.graph_lists[idx], self.graph_labels[idx]




class MoleculeBBBPDGL(torch.utils.data.Dataset):
    def __init__(self, data_dir, split, graphs, labels, num_graphs=None):
        self.data_dir = data_dir
        self.split = split

        # 加载数据
        df = pd.read_csv('%s.csv' % (data_dir))

        # 提取标签
        labels = df['p_np']

        # 提取分子 scaffold
        df['scaffold'] = df['smiles'].apply(self._get_scaffold)

        # 按 scaffold 分组
        scaffold_to_indices = defaultdict(list)
        for i, scaffold in enumerate(df['scaffold']):
            scaffold_to_indices[scaffold].append(i)

        # 按每个 scaffold 中分子数量递减排序
        sorted_scaffolds = sorted(scaffold_to_indices.items(), key=lambda x: len(x[1]), reverse=True)

        # 计算总分子数量
        total_samples = len(df)
        train_samples = int(0.8 * total_samples)
        val_samples = int(0.1 * total_samples)
        test_samples = total_samples - train_samples - val_samples

        # 将 scaffold 分组随机打乱
        shuffled_scaffolds = shuffle(sorted_scaffolds, random_state=42)

        # 拆分数据集，按每个 scaffold 的大小递减顺序划分
        train_scaffolds, val_test_scaffolds = [], []
        current_count = 0

        # 分配 scaffold 到训练集
        for scaffold, indices in shuffled_scaffolds:
            if current_count + len(indices) <= train_samples:
                train_scaffolds.append(scaffold)
                current_count += len(indices)
            else:
                val_test_scaffolds.append(scaffold)

        # 按照 scaffold 获取训练集的索引
        train_indices = [idx for scaffold in train_scaffolds for idx in scaffold_to_indices[scaffold]]

        # 获取验证集和测试集的索引
        val_test_indices = [idx for scaffold in val_test_scaffolds for idx in scaffold_to_indices[scaffold]]
        val_test_indices = np.array(val_test_indices)

        # 随机打乱验证集和测试集的索引
        val_test_indices = shuffle(val_test_indices, random_state=42)

        # 按比例划分验证集和测试集
        val_indices = val_test_indices[:val_samples]
        test_indices = val_test_indices[val_samples:]

        # 确保 indices 是 numpy 数组
        train_indices = np.array(train_indices)
        val_indices = np.array(val_indices)
        test_indices = np.array(test_indices)

        # 转换为 numpy 数组
        graphs = np.array(graphs)
        labels = np.array(labels)

        # 根据 split 构建数据集
        if split == 'train':
            self.graph_lists = graphs[train_indices].tolist()
            self.graph_labels = labels[train_indices].tolist()
            self.n_samples = len(train_indices)
        elif split == 'val':
            self.graph_lists = graphs[val_indices].tolist()
            self.graph_labels = labels[val_indices].tolist()
            self.n_samples = len(val_indices)
        elif split == 'test':
            self.graph_lists = graphs[test_indices].tolist()
            self.graph_labels = labels[test_indices].tolist()
            self.n_samples = len(test_indices)
        else:
            raise ValueError(f"Unknown split type: {split}")



    @staticmethod
    def _get_scaffold(smiles):
        """
        使用 RDKit 提取分子的 Scaffold
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return MurckoScaffold.MurckoScaffoldSmiles(mol=mol)    
    
    def _prepare(self):
        print("preparing %d graphs for the %s set..." % (self.num_graphs, self.split.upper()))
        
        for molecule in self.graphs:
            node_features = molecule['atom_type'].long()
            
            adj = molecule['bond_type']
            edge_list = (adj != 0).nonzero()  # converting adj matrix to edge_list
            
            edge_idxs_in_adj = edge_list.split(1, dim=1)
            edge_features = adj[edge_idxs_in_adj].reshape(-1).long()
            
            # Create the DGL Graph
            g = dgl.DGLGraph()
            g.add_nodes(molecule['num_atom'])
            g.ndata['feat'] = node_features
            
            for src, dst in edge_list:
                g.add_edges(src.item(), dst.item())
            g.edata['feat'] = edge_features
            
            self.graph_lists.append(g)
            self.graph_labels.append(molecule['logP_SA_cycle_normalized'])
        
    def __len__(self):
        """Return the number of graphs in the dataset."""
        return self.n_samples

    def __getitem__(self, idx):
        """
            Get the idx^th sample.
            Parameters
            ---------
            idx : int
                The sample index.
            Returns
            -------
            (dgl.DGLGraph, int)
                DGLGraph with node feature stored in `feat` field
                And its label.
        """
        return self.graph_lists[idx], self.graph_labels[idx]



class MoleculeSiderDGL(torch.utils.data.Dataset):
    def __init__(self, data_dir, split, graphs, labels, num_graphs=None):
        self.data_dir = data_dir
        self.split = split
        self.num_graphs = num_graphs

        df = pd.read_csv('%s.csv' % (data_dir))

        labels = df.iloc[:, 1:28].values

         # 随机划分数据集，按 8:1:1 的比例划分训练集、验证集、测试集
        train_indices, temp_indices = train_test_split(df.index, test_size=0.2, random_state=42)
        val_indices, test_indices = train_test_split(temp_indices, test_size=0.5, random_state=42)

        # 确保 indices 是 numpy 数组
        train_indices = np.array(train_indices)
        val_indices = np.array(val_indices)
        test_indices = np.array(test_indices)


        train_indices = np.array(train_indices)
        val_indices = np.array(val_indices)
        test_indices = np.array(test_indices)

        graphs = np.array(graphs)
        labels = np.array(labels)
        if split == 'train':
            self.graph_lists = graphs[train_indices].tolist()
            self.graph_labels = labels[train_indices].tolist()
            self.n_samples = len(train_indices)
        elif split == 'val':
            self.graph_lists = graphs[val_indices].tolist()
            self.graph_labels = labels[val_indices].tolist()
            self.n_samples = len(val_indices)
        elif split == 'test':
            self.graph_lists = graphs[test_indices].tolist()
            self.graph_labels = labels[test_indices].tolist()
            self.n_samples = len(test_indices)
        else:
            print(f"Error processing {split}:")

                    
        #self._prepare()

        
    def _prepare(self):
        print("preparing %d graphs for the %s set..." % (self.num_graphs, self.split.upper()))
        
        for molecule in self.graphs:
            node_features = molecule['atom_type'].long()
            
            adj = molecule['bond_type']
            edge_list = (adj != 0).nonzero()  # converting adj matrix to edge_list
            
            edge_idxs_in_adj = edge_list.split(1, dim=1)
            edge_features = adj[edge_idxs_in_adj].reshape(-1).long()
            
            # Create the DGL Graph
            g = dgl.DGLGraph()
            g.add_nodes(molecule['num_atom'])
            g.ndata['feat'] = node_features
            
            for src, dst in edge_list:
                g.add_edges(src.item(), dst.item())
            g.edata['feat'] = edge_features
            
            self.graph_lists.append(g)
            self.graph_labels.append(molecule['logP_SA_cycle_normalized'])
        
    def __len__(self):
        """Return the number of graphs in the dataset."""
        return self.n_samples

    def __getitem__(self, idx):
        """
            Get the idx^th sample.
            Parameters
            ---------
            idx : int
                The sample index.
            Returns
            -------
            (dgl.DGLGraph, int)
                DGLGraph with node feature stored in `feat` field
                And its label.
        """
        return self.graph_lists[idx], self.graph_labels[idx]

class MoleculeBACEDGL(torch.utils.data.Dataset):
    def __init__(self, data_dir, split, graphs, labels, num_graphs=None):
        self.data_dir = data_dir
        self.split = split

        # 加载数据
        df = pd.read_csv('%s.csv' % (data_dir))

        # 提取标签
        labels = df['Class']

        # 提取分子 scaffold
        df['scaffold'] = df['mol'].apply(self._get_scaffold)

        # 按 scaffold 分组
        scaffold_to_indices = defaultdict(list)
        for i, scaffold in enumerate(df['scaffold']):
            scaffold_to_indices[scaffold].append(i)

        # 按每个 scaffold 中分子数量递减排序
        sorted_scaffolds = sorted(scaffold_to_indices.items(), key=lambda x: len(x[1]), reverse=True)

        # 计算总分子数量
        total_samples = len(df)
        train_samples = int(0.8 * total_samples)
        val_samples = int(0.1 * total_samples)
        test_samples = total_samples - train_samples - val_samples

        # 拆分数据集，按每个 scaffold 的大小递减顺序划分
        train_scaffolds, val_test_scaffolds = [], []
        current_count = 0

        # 分配 scaffold 到训练集
        for scaffold, indices in sorted_scaffolds:
            if current_count + len(indices) <= train_samples:
                train_scaffolds.append(scaffold)
                current_count += len(indices)
            else:
                val_test_scaffolds.append(scaffold)

        # 按照 scaffold 获取训练集的索引
        train_indices = [idx for scaffold in train_scaffolds for idx in scaffold_to_indices[scaffold]]

        # 获取验证集和测试集的索引
        val_test_indices = [idx for scaffold in val_test_scaffolds for idx in scaffold_to_indices[scaffold]]
        val_test_indices = np.array(val_test_indices)

        # 随机打乱验证集和测试集的索引
        val_test_indices = shuffle(val_test_indices, random_state=42)

        # 按比例划分验证集和测试集
        val_indices = val_test_indices[:val_samples]
        test_indices = val_test_indices[val_samples:]

        # 确保 indices 是 numpy 数组
        train_indices = np.array(train_indices)
        val_indices = np.array(val_indices)
        test_indices = np.array(test_indices)

        # 转换为 numpy 数组
        graphs = np.array(graphs)
        labels = np.array(labels)

        # 根据 split 构建数据集
        if split == 'train':
            self.graph_lists = graphs[train_indices].tolist()
            self.graph_labels = labels[train_indices].tolist()
            self.n_samples = len(train_indices)
        elif split == 'val':
            self.graph_lists = graphs[val_indices].tolist()
            self.graph_labels = labels[val_indices].tolist()
            self.n_samples = len(val_indices)
        elif split == 'test':
            self.graph_lists = graphs[test_indices].tolist()
            self.graph_labels = labels[test_indices].tolist()
            self.n_samples = len(test_indices)
        else:
            raise ValueError(f"Unknown split type: {split}")


    @staticmethod
    def _get_scaffold(smiles):
        """
        使用 RDKit 提取分子的 Scaffold
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return MurckoScaffold.MurckoScaffoldSmiles(mol=mol)    
    
    def _prepare(self):
        print("preparing %d graphs for the %s set..." % (self.num_graphs, self.split.upper()))
        
        for molecule in self.graphs:
            node_features = molecule['atom_type'].long()
            
            adj = molecule['bond_type']
            edge_list = (adj != 0).nonzero()  # converting adj matrix to edge_list
            
            edge_idxs_in_adj = edge_list.split(1, dim=1)
            edge_features = adj[edge_idxs_in_adj].reshape(-1).long()
            
            # Create the DGL Graph
            g = dgl.DGLGraph()
            g.add_nodes(molecule['num_atom'])
            g.ndata['feat'] = node_features
            
            for src, dst in edge_list:
                g.add_edges(src.item(), dst.item())
            g.edata['feat'] = edge_features
            
            self.graph_lists.append(g)
            self.graph_labels.append(molecule['logP_SA_cycle_normalized'])
        
    def __len__(self):
        """Return the number of graphs in the dataset."""
        return self.n_samples

    def __getitem__(self, idx):
        """
            Get the idx^th sample.
            Parameters
            ---------
            idx : int
                The sample index.
            Returns
            -------
            (dgl.DGLGraph, int)
                DGLGraph with node feature stored in `feat` field
                And its label.
        """
        return self.graph_lists[idx], self.graph_labels[idx]

class MoleculeTOX21DGL(torch.utils.data.Dataset):
    def __init__(self, data_dir, split, graphs, labels, num_graphs=None):
        self.data_dir = data_dir
        self.split = split
        self.num_graphs = num_graphs

        df = pd.read_csv('%s.csv' % (data_dir))

        labels = df.iloc[:, 0:12].values

         # 随机划分数据集，按 8:1:1 的比例划分训练集、验证集、测试集
        train_indices, temp_indices = train_test_split(df.index, test_size=0.2, random_state=42)
        val_indices, test_indices = train_test_split(temp_indices, test_size=0.5, random_state=42)

        # 确保 indices 是 numpy 数组
        train_indices = np.array(train_indices)
        val_indices = np.array(val_indices)
        test_indices = np.array(test_indices)

        train_indices = np.array(train_indices)
        val_indices = np.array(val_indices)
        test_indices = np.array(test_indices)

        graphs = np.array(graphs)
        labels = np.array(labels)
        if split == 'train':
            self.graph_lists = graphs[train_indices].tolist()
            self.graph_labels = labels[train_indices].tolist()
            self.n_samples = len(train_indices)
        elif split == 'val':
            self.graph_lists = graphs[val_indices].tolist()
            self.graph_labels = labels[val_indices].tolist()
            self.n_samples = len(val_indices)
        elif split == 'test':
            self.graph_lists = graphs[test_indices].tolist()
            self.graph_labels = labels[test_indices].tolist()
            self.n_samples = len(test_indices)
        else:
            print(f"Error processing {split}:")

                    
        #self._prepare()

        
    def _prepare(self):
        print("preparing %d graphs for the %s set..." % (self.num_graphs, self.split.upper()))
        
        for molecule in self.graphs:
            node_features = molecule['atom_type'].long()
            
            adj = molecule['bond_type']
            edge_list = (adj != 0).nonzero()  # converting adj matrix to edge_list
            
            edge_idxs_in_adj = edge_list.split(1, dim=1)
            edge_features = adj[edge_idxs_in_adj].reshape(-1).long()
            
            # Create the DGL Graph
            g = dgl.DGLGraph()
            g.add_nodes(molecule['num_atom'])
            g.ndata['feat'] = node_features
            
            for src, dst in edge_list:
                g.add_edges(src.item(), dst.item())
            g.edata['feat'] = edge_features
            
            self.graph_lists.append(g)
            self.graph_labels.append(molecule['logP_SA_cycle_normalized'])
        
    def __len__(self):
        """Return the number of graphs in the dataset."""
        return self.n_samples

    def __getitem__(self, idx):
        """
            Get the idx^th sample.
            Parameters
            ---------
            idx : int
                The sample index.
            Returns
            -------
            (dgl.DGLGraph, int)
                DGLGraph with node feature stored in `feat` field
                And its label.
        """
        return self.graph_lists[idx], self.graph_labels[idx]


class MoleculeDGL(torch.utils.data.Dataset):
    def __init__(self, data_dir, split, num_graphs=None):
        self.data_dir = data_dir
        self.split = split
        self.num_graphs = num_graphs
        
        with open(data_dir + "/%s.pickle" % self.split,"rb") as f:
            self.data = pickle.load(f)

        if self.num_graphs in [10000, 1000]:
            # loading the sampled indices from file ./zinc_molecules/<split>.index
            with open(data_dir + "/%s.index" % self.split,"r") as f:
                data_idx = [list(map(int, idx)) for idx in csv.reader(f)]
                self.data = [ self.data[i] for i in data_idx[0] ]

            assert len(self.data)==num_graphs, "Sample num_graphs again; available idx: train/val/test => 10k/1k/1k"
        
        """
        data is a list of Molecule dict objects with following attributes
        
          molecule = data[idx]
        ; molecule['num_atom'] : nb of atoms, an integer (N)
        ; molecule['atom_type'] : tensor of size N, each element is an atom type, an integer between 0 and num_atom_type
        ; molecule['bond_type'] : tensor of size N x N, each element is a bond type, an integer between 0 and num_bond_type
        ; molecule['logP_SA_cycle_normalized'] : the chemical property to regress, a float variable
        """
        
        self.graph_lists = []
        self.graph_labels = []
        self.n_samples = len(self.data)
        self._prepare()
    
    def _prepare(self):
        print("preparing %d graphs for the %s set..." % (self.num_graphs, self.split.upper()))
        
        for molecule in self.data:
            node_features = molecule['atom_type'].long()
            
            adj = molecule['bond_type']
            edge_list = (adj != 0).nonzero()  # converting adj matrix to edge_list
            
            edge_idxs_in_adj = edge_list.split(1, dim=1)
            edge_features = adj[edge_idxs_in_adj].reshape(-1).long()
            
            # Create the DGL Graph
            g = dgl.DGLGraph()
            g.add_nodes(molecule['num_atom'])
            g.ndata['feat'] = node_features
            
            for src, dst in edge_list:
                g.add_edges(src.item(), dst.item())
            g.edata['feat'] = edge_features
            
            self.graph_lists.append(g)
            self.graph_labels.append(molecule['logP_SA_cycle_normalized'])
        
    def __len__(self):
        """Return the number of graphs in the dataset."""
        return self.n_samples

    def __getitem__(self, idx):
        """
            Get the idx^th sample.
            Parameters
            ---------
            idx : int
                The sample index.
            Returns
            -------
            (dgl.DGLGraph, int)
                DGLGraph with node feature stored in `feat` field
                And its label.
        """
        return self.graph_lists[idx], self.graph_labels[idx]
    
    
class MoleculeAqSolDGL(torch.utils.data.Dataset):
    def __init__(self, data_dir, split, num_graphs=None):
        self.data_dir = data_dir
        self.split = split
        self.num_graphs = num_graphs
        
        with open(data_dir + "/%s.pickle" % self.split,"rb") as f:
            self.data = pickle.load(f)
        
        """
        data is a list of tuple objects with following elements
        graph_object = (node_feat, edge_feat, edge_index, solubility)  
        """
        self.graph_lists = []
        self.graph_labels = []
        self.n_samples = len(self.data)
        assert num_graphs == self.n_samples
        
        self._prepare()
    
    def _prepare(self):
        print("preparing %d graphs for the %s set..." % (self.num_graphs, self.split.upper()))
        
        count_filter1, count_filter2 = 0,0
        
        for molecule in self.data:
            node_features = torch.LongTensor(molecule[0])
            edge_features = torch.LongTensor(molecule[1])
            
            # Create the DGL Graph
            g = dgl.graph((molecule[2][0], molecule[2][1]))
                        
            if g.num_nodes() == 0:
                count_filter1 += 1
                continue # skipping graphs with no bonds/edges
            
            if g.num_nodes() != len(node_features):
                count_filter2 += 1
                continue # cleaning <10 graphs with this discrepancy
            
            
            g.edata['feat'] = edge_features    
            g.ndata['feat'] = node_features
           
            self.graph_lists.append(g)
            self.graph_labels.append(torch.Tensor([molecule[3]]))
        print("Filtered graphs type 1/2: ", count_filter1, count_filter2)
        print("Filtered graphs: ", self.n_samples - len(self.graph_lists))
        
    def __len__(self):
        """Return the number of graphs in the dataset."""
        return len(self.graph_lists)

    def __getitem__(self, idx):
        return self.graph_lists[idx], self.graph_labels[idx]
    
    
class MoleculeDatasetDGL(torch.utils.data.Dataset):
    def __init__(self, name='Zinc'):
        t0 = time.time()
        self.name = name
        
        if self.name == 'AqSol':
            self.num_atom_type = 65 # known meta-info about the AqSol dataset; can be calculated as well 
            self.num_bond_type = 5 # known meta-info about the AqSol dataset; can be calculated as well
        else:            
            self.num_atom_type = 28 # known meta-info about the zinc dataset; can be calculated as well
            self.num_bond_type = 4 # known meta-info about the zinc dataset; can be calculated as well
        
        data_dir='./data/molecules'
        
        if self.name == 'ZINC-full':
            data_dir='./data/molecules/zinc_full'
            self.train = MoleculeDGL(data_dir, 'train', num_graphs=220011)
            self.val = MoleculeDGL(data_dir, 'val', num_graphs=24445)
            self.test = MoleculeDGL(data_dir, 'test', num_graphs=5000)
        elif self.name == 'ZINC':            
            self.train = MoleculeDGL(data_dir, 'train', num_graphs=10000)
            self.val = MoleculeDGL(data_dir, 'val', num_graphs=1000)
            self.test = MoleculeDGL(data_dir, 'test', num_graphs=1000)
        
        elif self.name == 'TOX21':
            data_dir = '/home/zsy123/benchmarking-gnns-master/data/molecules/TOX21/tox21_cleaned'
            df = pd.read_csv('%s.csv' % (data_dir))
            graphs = df['smiles'].apply(self.create_graph_from_smiles).tolist()
            labels = torch.tensor(df.iloc[:, 0:12].values)
            self.train = MoleculeTOX21DGL(data_dir, 'train', graphs, labels)
            self.val = MoleculeTOX21DGL(data_dir, 'val', graphs, labels)
            self.test = MoleculeTOX21DGL(data_dir, 'test', graphs, labels)
        
        elif self.name == 'BACE':
            data_dir = '/home/zsy123/benchmarking-gnns-master/data/molecules/BACE/bace'
            df = pd.read_csv('%s.csv' % (data_dir))
            graphs = df['mol'].apply(self.create_graph_from_smiles).tolist()
            labels = torch.tensor(df['Class']).tolist()
            self.train = MoleculeBACEDGL(data_dir, 'train', graphs, labels)
            self.val = MoleculeBACEDGL(data_dir, 'val', graphs, labels)
            self.test = MoleculeBACEDGL(data_dir, 'test', graphs, labels)

        elif self.name == 'BBBP':
            data_dir = '/home/zsy123/benchmarking-gnns-master/data/molecules/BBBP/BBBP_valid'
            df = pd.read_csv('%s.csv' % (data_dir))
            graphs = df['smiles'].apply(self.create_graph_from_smiles).tolist()
            labels = torch.tensor(df['p_np']).tolist()
            self.train = MoleculeBBBPDGL(data_dir, 'train', graphs, labels)
            self.val = MoleculeBBBPDGL(data_dir, 'val', graphs, labels)
            self.test = MoleculeBBBPDGL(data_dir, 'test', graphs, labels)  

        elif self.name == 'Sider':
            data_dir = '/home/zsy123/benchmarking-gnns-master/data/molecules/Sider/sider'
            df = pd.read_csv('%s.csv' % (data_dir))
            graphs = df['smiles'].apply(self.create_graph_from_smiles).tolist()
            labels = torch.tensor(df.iloc[:, 1:28].values)
            self.train = MoleculeSiderDGL(data_dir, 'train', graphs, labels)
            self.val = MoleculeSiderDGL(data_dir, 'val', graphs, labels)
            self.test = MoleculeSiderDGL(data_dir, 'test', graphs, labels)  

        elif self.name == 'ClinTox':
            data_dir = '/home/zsy123/benchmarking-gnns-master/data/molecules/ClinTox/clintox_valid'
            df = pd.read_csv('%s.csv' % (data_dir))
            graphs = df['smiles'].apply(self.create_graph_from_smiles).tolist()
            #labels = torch.tensor(df['FDA_APPROVED']).tolist()
            #labels = torch.tensor(df['CT_TOX']).tolist()

            labels = torch.tensor(df.iloc[:, 1:3].values)
            self.train = MoleculeClinToxDGL(data_dir, 'train', graphs, labels)
            self.val = MoleculeClinToxDGL(data_dir, 'val', graphs, labels)
            self.test = MoleculeClinToxDGL(data_dir, 'test', graphs, labels)   
        
        elif self.name == 'HIV':
            data_dir = '/home/zsy123/benchmarking-gnns-master/data/molecules/HIV/HIV_valid'
            df = pd.read_csv('%s.csv' % (data_dir))
            graphs = df['smiles'].apply(self.create_graph_from_smiles).tolist()
            labels = torch.tensor(df['HIV_active']).tolist()
            pos_enc_dim = 10  # 设定位置编码的维度
            for i, g in enumerate(graphs):
                graphs[i] = positional_encoding(g, pos_enc_dim)
            self.train = MoleculeHIVDGL(data_dir, 'train', graphs, labels)
            self.val = MoleculeHIVDGL(data_dir, 'val', graphs, labels)
            self.test = MoleculeHIVDGL(data_dir, 'test', graphs, labels)

        elif self.name == 'FreeSolv':
            data_dir = '/home/zsy123/benchmarking-gnns-master/data/molecules/FreeSolv/SAMPL'
            df = pd.read_csv('%s.csv' % (data_dir))
            graphs = df['smiles'].apply(self.create_graph_from_smiles).tolist()
            #labels = torch.tensor(df['FDA_APPROVED']).tolist()
            #labels = torch.tensor(df['CT_TOX']).tolist()

            labels = torch.tensor(df['expt']).tolist()
            self.train = MoleculeFreeSolvDGL(data_dir, 'train', graphs, labels)
            self.val = MoleculeFreeSolvDGL(data_dir, 'val', graphs, labels)
            self.test = MoleculeFreeSolvDGL(data_dir, 'test', graphs, labels)   

        elif self.name == 'Lipophilicity':
            data_dir = '/home/zsy123/benchmarking-gnns-master/data/molecules/Lipophilicity/Lipophilicity'
            df = pd.read_csv('%s.csv' % (data_dir))
            graphs = df['smiles'].apply(self.create_graph_from_smiles).tolist()
            #labels = torch.tensor(df['FDA_APPROVED']).tolist()
            #labels = torch.tensor(df['CT_TOX']).tolist()

            labels = torch.tensor(df['exp']).tolist()
            self.train = MoleculeLipophilicityDGL(data_dir, 'train', graphs, labels)
            self.val = MoleculeLipophilicityDGL(data_dir, 'val', graphs, labels)
            self.test = MoleculeLipophilicityDGL(data_dir, 'test', graphs, labels) 
        
        elif self.name == 'ESOL':
            data_dir = '/home/zsy123/benchmarking-gnns-master/data/molecules/ESOL/ESOL'
            df = pd.read_csv('%s.csv' % (data_dir))
            graphs = df['smiles'].apply(self.create_graph_from_smiles).tolist()
            #labels = torch.tensor(df['FDA_APPROVED']).tolist()
            #labels = torch.tensor(df['CT_TOX']).tolist()

            labels = torch.tensor(df['measured log solubility in mols per litre']).tolist()
            self.train = MoleculeESOLDGL(data_dir, 'train', graphs, labels)
            self.val = MoleculeESOLDGL(data_dir, 'val', graphs, labels)
            self.test = MoleculeESOLDGL(data_dir, 'test', graphs, labels) 

        elif self.name == 'QM8':
            data_dir = '/home/zsy123/benchmarking-gnns-master/data/molecules/QM8/QM8'
            df = pd.read_csv('%s.csv' % (data_dir))
            graphs = df['smiles'].apply(self.create_graph_from_smiles).tolist()
            #labels = torch.tensor(df['FDA_APPROVED']).tolist()
            #labels = torch.tensor(df['CT_TOX']).tolist()

            labels = torch.tensor(df.iloc[:, 1:13].values)
            self.train = MoleculeQM8DGL(data_dir, 'train', graphs, labels)
            self.val = MoleculeQM8DGL(data_dir, 'val', graphs, labels)
            self.test = MoleculeQM8DGL(data_dir, 'test', graphs, labels) 

        elif self.name == 'AqSol': 
            data_dir='./data/molecules/asqol_graph_raw'
            self.train = MoleculeAqSolDGL(data_dir, 'train', num_graphs=7985)
            self.val = MoleculeAqSolDGL(data_dir, 'val', num_graphs=998)
            self.test = MoleculeAqSolDGL(data_dir, 'test', num_graphs=999)
        print("Time taken: {:.4f}s".format(time.time()-t0))
    
    def create_graph_from_smiles(self,smiles):
        atom_featurizer = WeaveAtomFeaturizer(atom_data_field='feat')
        bond_featurizer = CanonicalBondFeaturizer(bond_data_field='feat', self_loop=False)
        try:
            # 使用 DGL-LifeSci 将 SMILES 转换为图
            # 将 smiles_to_bigraph 转换为部分函数，添加自环
            smiles_to_bigraph_with_self_loop = partial(smiles_to_bigraph, add_self_loop=False)
            
            # 使用 DGL-LifeSci 将 SMILES 转换为图
            graph = smiles_to_bigraph_with_self_loop(smiles, 
                                                    node_featurizer=atom_featurizer, 
                                                    edge_featurizer=bond_featurizer, 
                                                    canonical_atom_order=False)

            return graph
        except Exception as e:
            print(f"Error processing {smiles}: {str(e)}")
            return None    


def self_loop(g):
    """
        Utility function only, to be used only when necessary as per user self_loop flag
        : Overwriting the function dgl.transform.add_self_loop() to not miss ndata['feat'] and edata['feat']
        
        
        This function is called inside a function in MoleculeDataset class.
    """
    new_g = dgl.DGLGraph()
    new_g.add_nodes(g.number_of_nodes())
    new_g.ndata['feat'] = g.ndata['feat']
    
    src, dst = g.all_edges(order="eid")
    src = dgl.backend.zerocopy_to_numpy(src)
    dst = dgl.backend.zerocopy_to_numpy(dst)
    non_self_edges_idx = src != dst
    nodes = np.arange(g.number_of_nodes())
    new_g.add_edges(src[non_self_edges_idx], dst[non_self_edges_idx])
    new_g.add_edges(nodes, nodes)
    
    # This new edata is not used since this function gets called only for GCN, GAT
    # However, we need this for the generic requirement of ndata and edata
    new_g.edata['feat'] = torch.zeros(new_g.number_of_edges())
    return new_g



def positional_encoding(g, pos_enc_dim):
    """
        Graph positional encoding v/ Laplacian eigenvectors
        
        使用拉普拉斯特征向量生成图的位置信息编码

        参数：
        g: DGLGraph
            输入的图
        pos_enc_dim: int
            期望的位置信息编码维度

        返回：
        DGLGraph
            带有位置信息编码的图
    """
    # Laplacian
    A = g.adjacency_matrix(transpose=False, scipy_fmt="csr").astype(float)
    N = sp.diags(dgl.backend.asnumpy(g.in_degrees()).clip(1) ** -0.5, dtype=float)
    L = sp.eye(g.number_of_nodes()) - N * A * N

    # Eigenvectors with numpy
    # 使用 numpy 求解特征向量和特征值
    # L.toarray() 将稀疏矩阵 L 转换为密集矩阵格式
    # np.linalg.eig 计算矩阵的特征值（EigVal）和特征向量（EigVec）
    EigVal, EigVec = np.linalg.eig(L.toarray())
    # 对特征值进行排序，并相应地重排特征向量
    # idx 是特征值升序排序的索引数组
    idx = EigVal.argsort() # increasing order
    # 使用排序索引对特征值和特征向量进行重排
    EigVal, EigVec = EigVal[idx], np.real(EigVec[:,idx])
    # 选择特定数量的特征向量作为位置信息编码
    g.ndata['pos_enc'] = torch.from_numpy(EigVec[:,1:pos_enc_dim+1]).float() 

    # # Eigenvectors with scipy
    # EigVal, EigVec = sp.linalg.eigs(L, k=pos_enc_dim+1, which='SR')
    # EigVec = EigVec[:, EigVal.argsort()] # increasing order
    # g.ndata['pos_enc'] = torch.from_numpy(np.abs(EigVec[:,1:pos_enc_dim+1])).float() 
    
    n = g.number_of_nodes()
    # 如果节点数量小于或等于期望的位置信息编码维度
    if n <= pos_enc_dim:
        # 使用零填充位置信息编码，以匹配指定的维度
        # F.pad 在张量的最后一个维度上进行填充，(0, pos_enc_dim - n + 1) 指定填充的大小
        # value=float('0') 指定填充值为 0
        g.ndata['pos_enc'] = F.pad(g.ndata['pos_enc'], (0, pos_enc_dim - n + 1), value=float('0'))    
    return g


class MoleculeDataset(torch.utils.data.Dataset):

    def __init__(self, name, dataset_idx):
        """
        加载分子数据集
        参数:
            name: 数据集名称 (如 'TOX21', 'BACE', 等)
        """
        start = time.time()
        print(f"[I] 正在加载数据集 {name}...")
        self.name = name
        self.dataset_idx = dataset_idx  # 添加数据集索引

        # 数据集路径和名称配置
        dataset_paths = {
            'TOX21': ('tox21_8-1-1', '/home/zsy123/benchmarking-gnns-master/data/molecules/TOX21/'),
            'BACE': ('bace_8-1-1', '/home/zsy123/benchmarking-gnns-master/data/molecules/BACE/'),
            'BBBP': ('BBBP_8-1-1', '/home/zsy123/benchmarking-gnns-master/data/molecules/BBBP/'),
            'ClinTox': ('clintox_8-1-1', '/home/zsy123/benchmarking-gnns-master/data/molecules/ClinTox/'),
            'HIV': ('HIV_PE', '/home/zsy123/benchmarking-gnns-master/data/molecules/HIV/'),
            'Sider': ('sider_8-1-1', '/home/zsy123/benchmarking-gnns-master/data/molecules/Sider/'),
            'FreeSolv': ('FreeSolv_8-1-1', '/home/zsy123/benchmarking-gnns-master/data/molecules/FreeSolv/'),
            'Lipophilicity': ('Lipophilicity_8-1-1', '/home/zsy123/benchmarking-gnns-master/data/molecules/Lipophilicity/'),
            'ESOL': ('ESOL_8-1-1', '/home/zsy123/benchmarking-gnns-master/data/molecules/ESOL/'),
            'QM8': ('QM8_8-1-1', '/home/zsy123/benchmarking-gnns-master/data/molecules/QM8/')
        }

        if name not in dataset_paths:
            raise ValueError(f"不支持的数据集: {name}")

        dataset_name, data_dir = dataset_paths[name]

        # 检查文件路径是否存在
        dataset_file = os.path.join(data_dir, f"{dataset_name}.pkl")
        if not os.path.exists(dataset_file):
            raise FileNotFoundError(f"数据集文件未找到: {dataset_file}")

        # 加载数据集
        try:
            with open(dataset_file, "rb") as f:
                data = pickle.load(f)
                self.train = data[0]
                self.val = data[1]
                self.test = data[2]
                self.num_atom_type = data[3]
                self.num_bond_type = data[4]

            print(f"[I] 数据集 {name} 加载成功")
            print(f"训练集大小: {len(self.train)}, 测试集大小: {len(self.test)}, 验证集大小: {len(self.val)}")
        except Exception as e:
            raise RuntimeError(f"加载数据集 {name} 时发生错误: {e}")

        print(f"[I] 数据加载完成，耗时: {time.time() - start:.4f}s")


    # form a mini batch from a given list of samples = [(graph, label) pairs]
    def collate(self, samples):
        # The input samples is a list of pairs (graph, label).
        graphs, labels = map(list, zip(*samples))
        # print([len(label) for label in labels])  # 打印出所有标签的长度
        # print(labels)  # 查看标签的具体内容
        tensor_labels = [torch.tensor([item]) for item in labels]
        labels = torch.stack(tensor_labels).unsqueeze(-1)
        #labels = torch.tensor(np.array(labels)).unsqueeze(1)
        #tab_sizes_n = [ graphs[i].number_of_nodes() for i in range(len(graphs))]
        #tab_snorm_n = [ torch.FloatTensor(size,1).fill_(1./float(size)) for size in tab_sizes_n ]
        #snorm_n = torch.cat(tab_snorm_n).sqrt()  
        #tab_sizes_e = [ graphs[i].number_of_edges() for i in range(len(graphs))]
        #tab_snorm_e = [ torch.FloatTensor(size,1).fill_(1./float(size)) for size in tab_sizes_e ]
        #snorm_e = torch.cat(tab_snorm_e).sqrt()
        batched_graph = dgl.batch(graphs)       
        
        return batched_graph, labels, self.dataset_idx
    
    # prepare dense tensors for GNNs using them; such as RingGNN, 3WLGNN
    def collate_dense_gnn(self, samples, edge_feat):
        # The input samples is a list of pairs (graph, label).
        graphs, labels = map(list, zip(*samples))
        labels = torch.tensor(np.array(labels)).unsqueeze(1)
        #tab_sizes_n = [ graphs[i].number_of_nodes() for i in range(len(graphs))]
        #tab_snorm_n = [ torch.FloatTensor(size,1).fill_(1./float(size)) for size in tab_sizes_n ]
        #snorm_n = tab_snorm_n[0][0].sqrt()  
        
        #batched_graph = dgl.batch(graphs)
    
        g = graphs[0]
        adj = self._sym_normalize_adj(g.adjacency_matrix().to_dense())        
        """
            Adapted from https://github.com/leichen2018/Ring-GNN/
            Assigning node and edge feats::
            we have the adjacency matrix in R^{n x n}, the node features in R^{d_n} and edge features R^{d_e}.
            Then we build a zero-initialized tensor, say T, in R^{(1 + d_n + d_e) x n x n}. T[0, :, :] is the adjacency matrix.
            The diagonal T[1:1+d_n, i, i], i = 0 to n-1, store the node feature of node i. 
            The off diagonal T[1+d_n:, i, j] store edge features of edge(i, j).
        """

        zero_adj = torch.zeros_like(adj)

        if edge_feat:
            # use edge feats also to prepare adj
            adj_with_edge_feat = torch.stack([zero_adj for j in range(self.num_atom_type + self.num_bond_type)])
            adj_with_edge_feat = torch.cat([adj.unsqueeze(0), adj_with_edge_feat], dim=0)

            us, vs = g.edges()      
            for idx, edge_label in enumerate(g.edata['feat']):
                adj_with_edge_feat[edge_label.item()+1+self.num_atom_type][us[idx]][vs[idx]] = 1

            for node, node_label in enumerate(g.ndata['feat']):
                adj_with_edge_feat[node_label.item()+1][node][node] = 1
            
            x_with_edge_feat = adj_with_edge_feat.unsqueeze(0)
            
            return None, x_with_edge_feat, labels
        
        else:
            # use only node feats to prepare adj
            adj_no_edge_feat = torch.stack([zero_adj for j in range(self.num_atom_type)])
            adj_no_edge_feat = torch.cat([adj.unsqueeze(0), adj_no_edge_feat], dim=0)

            for node, node_label in enumerate(g.ndata['feat']):
                adj_no_edge_feat[node_label.item()+1][node][node] = 1

            x_no_edge_feat = adj_no_edge_feat.unsqueeze(0)

            return x_no_edge_feat, None, labels
    
    def _sym_normalize_adj(self, adj):
        deg = torch.sum(adj, dim = 0)#.squeeze()
        deg_inv = torch.where(deg>0, 1./torch.sqrt(deg), torch.zeros(deg.size()))
        deg_inv = torch.diag(deg_inv)
        return torch.mm(deg_inv, torch.mm(adj, deg_inv))
    
    def _add_self_loops(self):
        
        # function for adding self loops
        # this function will be called only if self_loop flag is True
            
        self.train.graph_lists = [self_loop(g) for g in self.train.graph_lists]
        self.val.graph_lists = [self_loop(g) for g in self.val.graph_lists]
        self.test.graph_lists = [self_loop(g) for g in self.test.graph_lists]

    def _add_positional_encodings(self, pos_enc_dim):
        
        # Graph positional encoding v/ Laplacian eigenvectors
        self.train.graph_lists = [positional_encoding(g, pos_enc_dim) for g in self.train.graph_lists]
        self.val.graph_lists = [positional_encoding(g, pos_enc_dim) for g in self.val.graph_lists]
        self.test.graph_lists = [positional_encoding(g, pos_enc_dim) for g in self.test.graph_lists]




