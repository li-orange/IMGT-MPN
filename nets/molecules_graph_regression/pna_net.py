import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from rdkit import Chem
from rdkit.Chem import Descriptors
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

import dgl

"""
    ResGatedGCN: Residual Gated Graph ConvNets
    An Experimental Study of Neural Networks for Variable Graphs (Xavier Bresson and Thomas Laurent, ICLR 2018)
    https://arxiv.org/pdf/1711.07553v2.pdf
"""
from dgl.nn import PNAConv
from layers.gated_gcn_layer import GatedGCNLayer
from layers.mlp_readout_layer import MLPReadout
from .model_utils import pair_atom_feats, unbatch_mask, unbatch_feats, Global_Reactivity_Attention, GELU


class PNANet(nn.Module):
    def __init__(self, net_params):
        super().__init__()
        in_atom_feat_dim = net_params['in_atom_feat_dim']
        in_bond_feat_dim = net_params['in_bond_feat_dim']
        num_atom_type = net_params['num_atom_type']
        num_bond_type = net_params['num_bond_type']
        hidden_dim = net_params['hidden_dim']
        out_dim = net_params['out_dim']
        in_feat_dropout = net_params['in_feat_dropout']
        dropout = net_params['dropout']
        n_layers = net_params['L']
        attention_heads = net_params['attention_heads']
        attention_layers = net_params['attention_layers']
        
        self.readout = net_params['readout']
        self.batch_norm = net_params['batch_norm']
        self.residual = net_params['residual']
        self.edge_feat = net_params['edge_feat']
        self.device = net_params['device']
        self.pos_enc = net_params['pos_enc']
        #使用gru
        self.use_gru = net_params['use_gru']
        #使用transfomer
        self.transformer = net_params['use_transformer']

        self.activation = GELU()
        self.att = Global_Reactivity_Attention(hidden_dim, attention_heads, attention_layers)
        self.linearB = nn.Linear(hidden_dim*2, hidden_dim)

        if self.pos_enc:
            pos_enc_dim = net_params['pos_enc_dim']
            self.embedding_pos_enc = nn.Linear(pos_enc_dim, hidden_dim)
        
        #self.embedding_h = nn.Embedding(in_atom_feat_dim, hidden_dim)
        self.embedding_h = nn.Linear(in_atom_feat_dim, hidden_dim)

        if self.edge_feat:
            #self.embedding_e = nn.Embedding(in_bond_feat_dim, hidden_dim)
            self.embedding_e = nn.Linear(in_bond_feat_dim, hidden_dim)
        else:
            self.embedding_e = nn.Linear(1, hidden_dim)
        
        self.in_feat_dropout = nn.Dropout(in_feat_dropout)
        
        self.layers = nn.ModuleList([ PNAConv(hidden_dim, hidden_dim, ['mean', 'max', 'min', 'std'], ['identity', 'amplification', 'attenuation'], 2.5, 
                                              dropout, edge_feat_size = hidden_dim ) for _ in range(n_layers-1) ]) 
        self.layers.append(PNAConv(hidden_dim, out_dim, ['mean', 'max', 'sum'], ['identity', 'amplification'], 2.5, 
                                              dropout, edge_feat_size = hidden_dim))
        #self.layers = nn.ModuleList([ GatedGCNLayer(hidden_dim, hidden_dim, dropout,
        #                                            self.batch_norm, self.residual) for _ in range(n_layers-1) ]) 
        #self.layers.append(GatedGCNLayer(hidden_dim, out_dim, dropout, self.batch_norm, self.residual))
        
        if self.readout == "S2S": 
            self.MLP_layer = MLPReadout(out_dim*2, 2)   # 2 out dim since classification problem
        else:
            self.MLP_layer = MLPReadout(out_dim, 2)   # 2 out dim since classification problem  
                
        #使用gru和S2S
        if self.use_gru:
            self.gru = nn.GRU(hidden_dim, hidden_dim)
        self.S2S = dgl.nn.pytorch.glob.Set2Set(hidden_dim, 1, 1)
            
    def forward(self, g, h, e, h_pos_enc=None):

        # input embedding
        #h = h.long()
        h = self.embedding_h(h)
        h = self.in_feat_dropout(h)
        if self.pos_enc:
            h_pos_enc = self.embedding_pos_enc(h_pos_enc.float()) 
            h = h + h_pos_enc
        if not self.edge_feat: # edge feature set to 1
            e = torch.ones(e.size(0),1).to(self.device)
        e = self.embedding_e(e)   
        
        # convnets
        for conv in self.layers:
            h_t = conv(g, h, e)
            
            if self.use_gru:
                # Use GRU
                h_t = h_t.unsqueeze(0)
                h = h.unsqueeze(0)
                h = self.gru(h, h_t)[1]

                # Recover back in regular form
                h = h.squeeze()
            else:
                h = h_t
                
        if self.transformer:
            atom_feats = h
            bond_feats = self.linearB(pair_atom_feats(g, atom_feats))
            edit_feats, mask = unbatch_mask(g, atom_feats, bond_feats)
            attention_score, edit_feats = self.att(edit_feats, mask)
            atom_feats, bond_feats = unbatch_feats(g, edit_feats)
            g.ndata['h'] = atom_feats
        else:
            g.ndata['h'] = h
               
        if self.readout == "sum":
            hg = dgl.sum_nodes(g, 'h')
        elif self.readout == "max":
            hg = dgl.max_nodes(g, 'h')
        elif self.readout == "mean":
            hg = dgl.mean_nodes(g, 'h')
        elif self.readout == "mean+max":
            hg = dgl.mean_nodes(g, 'h') + dgl.max_nodes(g, 'h')
        elif self.readout == "S2S":    
            # Set2Set Readout for Graph Tasks
            if self.transformer:
                hg = self.S2S(g, atom_feats)
            else:
                hg = self.S2S(g, h)
        else:
            hg = dgl.mean_nodes(g, 'h')  # default readout is mean nodes
            
        return self.MLP_layer(hg)
        
    def loss(self, scores, targets):
        # loss = nn.MSELoss()(scores,targets)
        #使用交叉损失熵
        # 确保 targets 是长整型
        #targets = targets.float()
        loss = nn.CrossEntropyLoss()(scores, targets)
        #loss = nn.L1Loss()(scores, targets)
        return loss
