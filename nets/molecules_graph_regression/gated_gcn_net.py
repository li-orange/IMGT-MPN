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
import matplotlib
matplotlib.use('Agg')

import dgl

"""
    ResGatedGCN: Residual Gated Graph ConvNets
    An Experimental Study of Neural Networks for Variable Graphs (Xavier Bresson and Thomas Laurent, ICLR 2018)
    https://arxiv.org/pdf/1711.07553v2.pdf
"""
from layers.gated_gcn_layer import GatedGCNLayer
from layers.mlp_readout_layer import MLPReadout
from .model_utils import pair_atom_feats, unbatch_mask, unbatch_feats, extend_batch_graph, Global_Reactivity_Attention, Global_Reactivity_Attention_MoE, Global_Reactivity_Attention_MoE_GNN, Global_Reactivity_Attention_GNN, GatingNetwork, GELU , batch_shortest_path_matrices, get_bond_feats_matrices, compute_node_degrees, visualize_attention_matrices, visualize_attention_molecules
import os

class GatedGCNNet(nn.Module):
    def __init__(self, net_params):
        super().__init__()
        in_atom_feat_dim = net_params['in_atom_feat_dim']
        in_bond_feat_dim = net_params['in_bond_feat_dim']
        #num_atom_type = net_params['num_atom_type']
        #num_bond_type = net_params['num_bond_type']
        hidden_dim = net_params['hidden_dim']
        out_dim = net_params['out_dim']
        in_feat_dropout = net_params['in_feat_dropout']
        dropout = net_params['dropout']
        n_layers = net_params['L']
        attention_heads = net_params['attention_heads']
        attention_layers = net_params['attention_layers']
        num_experts = net_params['num_experts']
        num_experts_per_tok = net_params['num_experts_per_tok']
        
        self.readout = net_params['readout']
        self.batch_norm = net_params['batch_norm']
        self.residual = net_params['residual']
        self.edge_feat = net_params['edge_feat']
        self.device = net_params['device']
        self.pos_enc = net_params['pos_enc']
        #使用gru
        self.use_gru = net_params['use_gru']
        #使用transfomer
        self.use_transformer = net_params['use_transformer']
        #使用MoE
        self.use_moe = net_params['use_moe']

        self.task_type = net_params['task_type']
        self.num_tasks = net_params['num_tasks']
        self.activation = GELU()

        #self.att = Global_Reactivity_Attention(hidden_dim, attention_heads, attention_layers)
        #self.att_MoE = Global_Reactivity_Attention_MoE(hidden_dim, attention_heads, attention_layers, 
        #                                               num_experts=4, num_experts_per_tok=2,  device=self.device)
        self.att_MoE_GNN = Global_Reactivity_Attention_MoE_GNN(hidden_dim, attention_heads, self.batch_norm, self.residual, 
                                                               n_layers=6, num_experts=4, num_experts_per_tok=2,  device=self.device)
        #self.att_GNN = Global_Reactivity_Attention_GNN(hidden_dim, attention_heads, self.batch_norm, self.residual, n_layers=6)
        

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
        
        self.layers = nn.ModuleList([ GatedGCNLayer(hidden_dim, hidden_dim, dropout, self.batch_norm, self.residual) for _ in range(n_layers-1) ]) 
        self.layers.append(GatedGCNLayer(hidden_dim, out_dim, dropout, self.batch_norm, self.residual))
        # 

        #regressor1: FreeSolv
        self.regressor1 = MLPReadout(out_dim*2, 1)    
        #regressor2: Lipophilicity
        self.regressor2 = MLPReadout(out_dim*2, 1)    
        #regressor3: ESOL
        self.regressor3 = MLPReadout(out_dim*2, 1)    
        #regressor4: QM8
        #self.regressor4 = MLPReadout(out_dim*2, 12)    
           

        if self.num_tasks[0] == 1:
            self.classifier_single = MLPReadout(out_dim*2, 2)
        else:
            self.classifier_single = MLPReadout(out_dim*2, self.num_tasks[0])

        #classifier1: BACE
        self.classifier1 = MLPReadout(out_dim*2, 2)
        #classifier2: BBBP  
        self.classifier2 = MLPReadout(out_dim*2, 2)
        #classifier3: ClinTox
        self.classifier3 = MLPReadout(out_dim*2, 2)
        #classifier4: Sider
        self.classifier4 = MLPReadout(out_dim*2, 27)
        #classifier5: Tox21
        self.classifier5 = MLPReadout(out_dim*2, 12)
        
        """
        # 使用Xavier初始化
        for m in [self.classifier1, self.classifier2, self.classifier3]:
            for layer in m.children():
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    nn.init.zeros_(layer.bias)
        """

        """
        if self.num_tasks == 1:
            self.MLP_layer = MLPReadout(out_dim*2, 2)   # 2 out dim since classification problem  
        else:
            self.MLP_layer = MLPReadout(out_dim*2, self.num_tasks)   # 2 out dim since classification problem
        """
          
        #使用gru和S2S
        if self.use_gru:
           self.gru = nn.GRU(hidden_dim, hidden_dim)
        # self.S2S = dgl.nn.pytorch.glob.Set2Set(hidden_dim, 1, 1)

        # 定义嵌入层：将最短路径长度映射到 8 维
        self.dis_embedding_layer = nn.Embedding(num_embeddings=500, embedding_dim=attention_heads)  # 假设最大路径长度不会超过500
        
        # 定义节点度数嵌入层
        #self.node_degrees_embedding_layer = nn.Embedding(num_embeddings=10, embedding_dim=hidden_dim)

        # 定义线性层，将边特征的64维特征映射为8维
        self.bond_att_layer = nn.Linear(hidden_dim, attention_heads)

        # 定义线性层，将拼接的64+64维特征映射为64维
        self.combined_layer = nn.Linear(hidden_dim*2, hidden_dim)

        # Batch normalization layers
        self.batch_norm = nn.BatchNorm1d(hidden_dim )

        # 数据集嵌入层
        #self.dataset_embedding = nn.Embedding(num_embeddings= len(self.num_tasks), embedding_dim=hidden_dim*2)

    def forward(self, dataset_idx, g, h, e, training_flag, h_pos_enc=None):

        #node_degrees = compute_node_degrees(g)
        #embedded_degrees = self.node_degrees_embedding_layer(node_degrees.long())

        #batch_size = g.batch_size

        # 扩展 dataset_idx 为 [16, 1] 形状
        #dataset_idx_emb = torch.tensor([dataset_idx] * batch_size).view(-1, 1).long().to(self.device)

        # 获取数据集的嵌入向量
        #dataset_emb = self.dataset_embedding(dataset_idx_emb).squeeze(1)
        
        h = self.embedding_h(h) #+ embedded_degrees
        
        h = self.in_feat_dropout(h)
        if self.pos_enc:
            nn.init.uniform_(self.embedding_pos_enc.weight, -0.1, 0.1)
            h_pos_enc = self.embedding_pos_enc(h_pos_enc.float()) 
            h = h + h_pos_enc
        if not self.edge_feat: # edge feature set to 1
            e = torch.ones(e.size(0),1).to(self.device)
        e = self.embedding_e(e)     

        h_residual = h
                                                  
        if False: #self.use_transformer:
            atom_feats = h
            bond_feats = e

            #bond_feats =  self.linearB( torch.cat((pair_atom_feats(g, atom_feats),e), dim = 1) )  #原子对信息和化学键信息拼接
            dis_matrices = batch_shortest_path_matrices(g).to(self.device)
            dis_matrices_bias = self.dis_embedding_layer(dis_matrices.long()).permute(0, 3, 1, 2) 
            
            bond_feats_matrices = get_bond_feats_matrices(g, bond_feats).to(self.device)
            bond_feats_matrices_bias = self.bond_att_layer(bond_feats_matrices).permute(0, 3, 1, 2) 
            
            # 使用 alpha 和 beta 的加权方式计算 att_bias
            att_bias = dis_matrices_bias + bond_feats_matrices_bias #


            edit_feats, mask = unbatch_mask(g, atom_feats, bond_feats)
            #att_bias = dis_matrices_bias + bond_feats_matrices_bias

            new_g = extend_batch_graph(g)
            attention_score, edit_feats, bond_feats = self.att_MoE_GNN(new_g, edit_feats, bond_feats, mask, att_bias)
            #attention_score, edit_feats = self.att(edit_feats, mask, att_bias)
            atom_feats = unbatch_feats(g, edit_feats)

            atom_feats = atom_feats + h_residual

            g.ndata['h'] = atom_feats
            g.edata['e'] = bond_feats
        else:
            g.ndata['h'] = h
            g.edata['e'] = e
        
        if True: #self.use_moe:
            atom_feats_gnn = h
            bond_feats_gnn = e
            # convnets
            for conv in self.layers:
                h_t, bond_feats_gnn = conv(g, atom_feats_gnn, bond_feats_gnn)
                
                if self.use_gru:
                    # Use GRU
                    h_t = h_t.unsqueeze(0)
                    atom_feats_gnn = atom_feats_gnn.unsqueeze(0)
                    atom_feats_gnn = self.gru(atom_feats_gnn, h_t)[1]

                    # Recover back in regular form
                    atom_feats_gnn = atom_feats_gnn.squeeze()
                else:
                    atom_feats_gnn = h_t
            
            g.ndata['h'] = atom_feats_gnn
            g.edata['e'] = bond_feats_gnn
            atom_feats = atom_feats_gnn
            attention_score = 0
               
        #if isinstance(attention_score, list):
        #    attention_score = torch.tensor(attention_score)
        if False :
            visualize_attention_matrices(attention_score, save_dir = "attention_matrices")
        if False:
            visualize_attention_molecules(attention_score, g, save_dir="/home/zsy123/benchmarking-gnns-master/attention_visualization")

        if self.readout == "sum":
            hg_node = dgl.sum_nodes(g, 'h')
            hg_edge = dgl.sum_edges(g, 'e')
            # hg = dgl.sum_nodes(g, 'h')
        elif self.readout == "max":
            hg_node = dgl.max_nodes(g, 'h')
            hg_edge = dgl.max_edges(g, 'e')
            # hg = dgl.max_nodes(g, 'h')
        elif self.readout == "mean":
            hg_node = dgl.mean_nodes(g, 'h')
            hg_edge = dgl.mean_edges(g, 'e')
            # hg = dgl.mean_nodes(g, 'h')
        elif self.readout == "mean+max":
            hg = dgl.mean_nodes(g, 'h') + dgl.max_nodes(g, 'h')
        elif self.readout == "mean+sum":
            hg_node = torch.cat([dgl.mean_nodes(g, 'h'), dgl.sum_nodes(g, 'h')], dim=-1)
            hg_edge = torch.cat([dgl.mean_edges(g, 'e'), dgl.sum_edges(g, 'e')], dim=-1)
        elif self.readout == "S2S":    
            # Set2Set Readout for Graph Tasks
            if self.transformer:
                hg_node = self.S2S(g, atom_feats)
            else:
                hg_node = self.S2S(g, h)
        else:
            hg_node = dgl.mean_nodes(g, 'h')  # default readout is mean nodes
            hg_edge = dgl.mean_edges(g, 'e')
        
        hg = torch.cat((hg_node, hg_edge), dim = 1)   #原子特征和边特征拼接

        # 拼接图特征与数据集嵌入
        combined_features = hg
        #combined_features = torch.cat([hg, dataset_emb], dim=-1)

        #return self.MLP_layer(hg)
        
        if self.task_type =='cls':
            
            if len(self.num_tasks) == 1:
                output = self.classifier_single(combined_features)
                #self.save_model_features(g, h, attention_score, file_name="/home/zsy123/benchmarking-gnns-master/nets/molecules_graph_regression/output_features_clintox.csv")
                if self.num_tasks[0] > 1 and not training_flag:
                    output = torch.sigmoid(output)  # 每个标签的预测概率
                return output, attention_score
            else:
                # 根据 dataset_idx 选择对应的分类器
                if dataset_idx == 0:  # 数据集1 BACE
                    output = self.classifier1(combined_features)

                elif dataset_idx == 1:  # 数据集2 BBBP
                    output = self.classifier2(combined_features)

                elif dataset_idx == 2:  # 数据集3 ClinTox
                    output = self.classifier3(combined_features)
                    if not training_flag:
                        output = torch.sigmoid(output)  # 每个标签的预测概率

                elif dataset_idx == 3:  # 数据集4 Sider
                    output = self.classifier4(combined_features)
                    if not training_flag:
                        output = torch.sigmoid(output)  # 每个标签的预测概率

                elif dataset_idx == 4:  # 数据集4 Tox21
                    output = self.classifier5(combined_features)
                    if not training_flag:
                        output = torch.sigmoid(output)  # 每个标签的预测概率
                    
                return output, attention_score
        else:
            if dataset_idx == 0:  # 数据集1 FreeSolv
                return self.regressor1(combined_features),attention_score
            elif dataset_idx == 1:  # 数据集1 Lipophilicity
                return self.regressor2(combined_features),attention_score
            elif dataset_idx == 2:  # 数据集1 ESOL
                return self.regressor3(combined_features),attention_score
            
        
        
        
    def loss(self, scores, targets, num_task):
        # loss = nn.MSELoss()(scores,targets)
        #使用交叉损失熵
        # 确保 targets 是长整型
        #targets = targets.float()
        #loss = nn.CrossEntropyLoss()(scores, targets)
        #loss = nn.L1Loss()(scores, targets)

        task_type = self.task_type
        num_tasks = self.num_tasks

        if task_type == 'cls':
            # 单标签多分类任务
            # CrossEntropyLoss 自动应用 softmax，不要在模型输出中进行 softmax
            if num_task == 1:
                if targets.dtype != torch.long:
                    targets = targets.long()
                loss_fn = nn.CrossEntropyLoss()
                loss = loss_fn(scores, targets)
            else:
                # 多标签分类任务
                # 使用 BCEWithLogitsLoss，用于每个标签二元分类
                if targets.dtype != torch.float:
                    targets = targets.float()
                loss_fn = nn.BCEWithLogitsLoss()
                loss = loss_fn(scores, targets)
        
        elif task_type == 'reg':
            # 回归任务，使用 MSELoss 或 L1Loss
            #loss_fn = nn.L1Loss()
            loss_fn = nn.MSELoss()
            loss = loss_fn(scores, targets)
            loss = torch.sqrt(loss)  # 计算均方根误差 (RMSE)
    
        return loss
    
    def save_model_features(self, g, h, attention_score, file_name="output_features.csv"):
        """
        保存模型的特征到文件。
        
        参数:
        dataset_idx (int): 数据集的索引，确定要保存的数据集（例如 ClinTox 数据集为 2）。
        g (DGLGraph): 图数据，包含节点和边的特征。
        h (Tensor): 节点特征。
        e (Tensor): 边特征。
        attention_score (Tensor): 注意力分数矩阵。
        file_name (str): 输出文件名，默认为 "output_features.csv"。
        """
        smiles = generate_smiles_from_graph(g)
        atom_features = g.ndata['h'].detach().cpu().numpy()  # 原子特征，使用 detach() 以防止梯度计算
        bond_features = g.edata['e'].detach().cpu().numpy()  # 键特征，使用 detach()
        graph_features = h.detach().cpu().numpy()  # 整图特征，使用 detach()

        attention_score = attention_score[-1]

        # 确保 attention_score 是一个张量，如果它是一个列表则转换为张量
        if isinstance(attention_score, list):
            attention_score = torch.tensor(attention_score)
        
        attention_scores = attention_score.detach().cpu().numpy()  # 注意力分数矩阵，使用 detach()

        # 将输出特征保存为文本文件或 CSV 文件
        output_data = []
        for i in range(len(smiles)):
            # 每个批次的 SMILES 和特征
            batch_smiles = smiles[i]
            batch_atom_features = atom_features[i]
            batch_bond_features = bond_features[i]
            batch_graph_features = graph_features[i]
            
            # 注意力矩阵：对每个注意力头和节点对的注意力矩阵进行处理
            for head_idx in range(attention_scores.shape[1]):  # num_attention_heads
                attention_matrix = attention_scores[i, head_idx]  # 获取当前注意力头的矩阵
                
                # 将注意力矩阵展平成列表形式，或者按需处理
                attention_matrix_flat = attention_matrix.flatten().tolist()
                
                # 将每个分子的 SMILES、特征和注意力矩阵保存
                output_data.append({
                    'SMILES': batch_smiles,
                    'Atom Features': batch_atom_features.tolist(),
                    'Bond Features': batch_bond_features.tolist(),
                    'Graph Features': batch_graph_features.tolist(),
                    'Attention Head': head_idx,  # 记录注意力头
                    'Attention Scores': attention_matrix_flat  # 展平后的注意力矩阵
                })

        # 将数据保存为 CSV 文件
        df = pd.DataFrame(output_data)
        df.to_csv(file_name, index=False)
        print(f"Features saved to {file_name}")
def generate_smiles_from_graph(bg, atom_type_map=None):
    smiles = []
    atom_types = [ 
            'H', 'C', 'N', 'O', 'F', 'P', 'S', 'Cl', 'Br', 'I',  # 从BBBP
            'Ca', 'Na', 'B',  # 也来自BBBP
            'Mn', 'Zn', 'Co', 'Fe', 'Tl', 'Hg', 'Bi', 'Se', 'Cu', 'Au', 'Pt', 'Al', 'Ti', 'Tc', 'Si', 'As', 'Cr',  # 从Clintox
            '*',  # Clintox中包含的未知原子

            # tox21 数据集中的元素
            'Yb', 'Mo', 'In', 'Li', 'Ni', 'Zr', 'Ge', 'Mg', 'Sn', 'Pb', 'K', 'Ba',

            # sider 数据集中的元素
            'Cf', 'Gd', 'La', 'Ra', 'Sr', 'Ag', 'Sm', 'Ga',
            
            # freeSolv
            'P', 'S', 'O', 'N'
            ]
    
    if atom_type_map is None:
        atom_type_map = {atom: idx for idx, atom in enumerate(atom_types)}

    # 获取图数据
    sg = bg.remove_self_loop()  # 删除自环
    gs = dgl.unbatch(sg)

    for batch_idx, g in enumerate(gs):

        # 获取图的节点数
        num_nodes = g.num_nodes()

        # 获取边列表
        src_nodes, dst_nodes = g.edges()

        # 创建 RDKit 分子对象
        mol = Chem.RWMol()

        # 获取节点的原子类型
        atom_types = g.ndata['atom_type'].detach().cpu().numpy()

        # 添加原子
        atom_idx_map = {}
        for i in range(num_nodes):
            atom_symbol = list(atom_type_map.keys())[list(atom_type_map.values()).index(atom_types[i])]
            atom = Chem.Atom(atom_symbol)
            atom_idx_map[i] = mol.AddAtom(atom)

        # 添加边（检查是否已存在该边）
        added_bonds = []
        for i, (u, v) in enumerate(zip(src_nodes, dst_nodes)):
            atom_u_idx = u.item()
            atom_v_idx = v.item()

            # 确保无向边顺序一致
            sorted_bond = tuple(sorted([atom_u_idx, atom_v_idx]))  # 将边的端点排序
            if sorted_bond not in added_bonds:  # 如果边未添加过
                mol.AddBond(atom_idx_map[atom_u_idx], atom_idx_map[atom_v_idx], Chem.BondType.SINGLE)
                added_bonds.append(sorted_bond)  # 添加到已添加的边列表中
        
        # 获取 SMILES 字符串
        smiles.append(Chem.MolToSmiles(mol)) 

    return smiles

