import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from typing import Optional
from torch import Tensor
from dgl.nn.pytorch import NNConv
import math
import numpy as np
import networkx as nx
import os
import matplotlib.pyplot as plt

from layers.gated_gcn_layer import GatedGCNLayer

from sklearn.preprocessing import MinMaxScaler
from rdkit import Chem
from rdkit.Chem import Draw
from PIL import Image, ImageDraw, ImageFont

def visualize_attention_molecules_baoliu(attention_scores, bg, save_dir=None, atom_type_map=None):
    """
    根据注意力矩阵和分子图批次可视化每个分子，并根据注意力分数渲染颜色。
    注意力矩阵形状为: [batch_size, num_attention_heads, num_nodes, num_nodes]
    
    Parameters:
    - attention_scores: 注意力矩阵，形状为 [batch_size, num_attention_heads, num_nodes, num_nodes]
    - bg: 批次中的图
    - save_dir: 可选，保存图像的目录
    - atom_type_map: 一个字典，将原子类型映射到序号（整数）
    """
    atom_types = [
        'H', 'C', 'N', 'O', 'F', 'P', 'S', 'Cl', 'Br', 'I',  # 从BBBP
        'Ca', 'Na', 'B',  # 也来自BBBP
        'Mn', 'Zn', 'Co', 'Fe', 'Tl', 'Hg', 'Bi', 'Se', 'Cu', 'Au', 'Pt', 'Al', 'Ti', 'Tc', 'Si', 'As', 'Cr',  # 从Clintox
        '*'  # Clintox中包含的未知原子
    ]

    # 通过enumerate生成原子符号到索引的映射
    if atom_type_map is None:
        atom_type_map = {atom: idx for idx, atom in enumerate(atom_types)}  # 默认值，如果没有传入

    # 为每个原子类型指定一个特定的颜色
    atom_color_map = {
        'H': '#FF0000',  # 红色
        'C': '#0000FF',  # 蓝色
        'N': '#00FF00',  # 绿色
        'O': '#FF4500',  # 橙色
        'F': '#800080',  # 紫色
        'P': '#FFD700',  # 金色
        'S': '#FFFF00',  # 黄色
        'Cl': '#008000',  # 深绿色
        'Br': '#D2691E',  # 巧克力色
        'I': '#800000',  # 棕色
        'Ca': '#8B4513',  # SaddleBrown
        'Na': '#000080',  # Navy
        'B': '#A52A2A',  # Brown
        'Mn': '#D3D3D3',  # LightGray
        'Zn': '#C0C0C0',  # Silver
        'Co': '#0000CD',  # MediumBlue
        'Fe': '#A52A2A',  # Brown
        'Tl': '#B22222',  # FireBrick
        'Hg': '#696969',  # DimGray
        'Bi': '#8B0000',  # DarkRed
        'Se': '#B8860B',  # DarkGoldenrod
        'Cu': '#BDB76B',  # DarkKhaki
        'Au': '#FFD700',  # 金色
        'Pt': '#D3D3D3',  # LightGray
        'Al': '#B0C4DE',  # LightSteelBlue
        'Ti': '#2F4F4F',  # DarkSlateGray
        'Tc': '#708090',  # SlateGray
        'Si': '#808080',  # Gray
        'As': '#D2691E',  # Chocolate
        'Cr': '#A9A9A9',  # DarkGray
        '*': '#A9A9A9'    # 未知原子，使用灰色
    }

    sg = bg.remove_self_loop()
    gs = dgl.unbatch(sg)
    
    # 遍历批次中的每个图
    attention_scores = attention_scores[-1]  # 使用最后一层的注意力分数
    for batch_idx, g in enumerate(gs):
        # 获取当前图的注意力矩阵（选择第一个注意力头）
        attention_matrix = attention_scores[batch_idx][0].detach().cpu().numpy()  # 选择第一个注意力头

        # 获取图的实际节点数
        num_nodes = g.num_nodes()

        # 截取注意力矩阵的有效部分 (去除填充部分)
        attention_matrix = attention_matrix[:num_nodes, :num_nodes]

        # 获取边列表，假设每个图的边列表是由两个 tensor 组成
        edge_list = g.edges()
        src_nodes, dst_nodes = edge_list  # 每个 tensor 表示边的源节点和目标节点

        # 注意力矩阵的每一行/列对应分子中的节点
        edge_attention_values = []

        # 遍历每条边，获取对应的注意力值
        for i, (u, v) in enumerate(zip(src_nodes, dst_nodes)):
            # u 和 v 是源节点和目标节点的索引，我们用它们从注意力矩阵中提取值
            if u.item() < num_nodes and v.item() < num_nodes:  # 确保索引有效
                edge_attention_values.append(attention_matrix[u.item(), v.item()])
            else:
                edge_attention_values.append(0)  # 如果索引无效，则将其设置为0

        # 检查 edge_attention_values 是否为空
        if len(edge_attention_values) == 0:
            print(f"Warning: No valid edge attention values for batch {batch_idx}. Skipping normalization.")
            continue  # 跳过该图

        # 归一化注意力值到 [0, 1]
        scaler = MinMaxScaler()
        edge_attention_values = np.array(edge_attention_values).reshape(-1, 1)
        edge_attention_values_normalized = scaler.fit_transform(edge_attention_values).flatten()

        # 创建一个 NetworkX 图
        G = nx.Graph()

        # 获取节点的原子类型（从 `ndata['atom_type']` 中获取）
        atom_types = g.ndata['atom_type'].detach().cpu().numpy()  # 假设这里存储的是原子类型的序号

        # 创建节点颜色列表，根据原子类型给节点上色
        node_colors = [atom_color_map[list(atom_type_map.keys())[list(atom_type_map.values()).index(atom_types[i])]] for i in range(num_nodes)]

        # 获取原子符号的列表，用于节点标签显示
        atom_symbols = [list(atom_type_map.keys())[list(atom_type_map.values()).index(atom_types[i])] for i in range(num_nodes)]

        # 添加节点
        for i in range(num_nodes):
            G.add_node(i)

        # 添加边并设置颜色
        edge_colors = []
        for i, (u, v) in enumerate(zip(src_nodes, dst_nodes)):
            edge_colors.append(plt.cm.viridis(edge_attention_values_normalized[i]))

        # 在 NetworkX 图中添加边，指定颜色和权重
        edges = [(src.item(), dst.item()) for src, dst in zip(src_nodes, dst_nodes)]
        G.add_edges_from(edges)

        # 绘制图形
        pos = nx.spring_layout(G, seed=42, k=0.2)  # 使用 spring_layout 来进行节点布局，并调整图形布局

        # 绘制节点（节点颜色根据原子类型来渲染）
        plt.figure(figsize=(10, 8))
        nx.draw(G, pos, node_size=700, node_color=node_colors, with_labels=True, font_weight='bold', font_size=12,
                edge_color=edge_colors, width=3, alpha=0.8, font_color='black', labels={i: atom_symbols[i] for i in range(num_nodes)})

        # 显示注意力的颜色条
        sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=min(edge_attention_values_normalized), vmax=max(edge_attention_values_normalized)))
        sm.set_array([])
        plt.colorbar(sm, label='Attention Value')

        # 图形标题
        plt.title(f'Molecular Graph with Attention Visualization - Batch {batch_idx}', fontsize=16)

        # 保存或显示图像
        if save_dir:
            plt.savefig(f"{save_dir}/graph_batch_{batch_idx}.png")
            #print(f"Graph {batch_idx} saved to {save_dir}/graph_batch_{batch_idx}.png")
        else:
            plt.show()
            
        plt.close()  # 关闭图形以释放内存


import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import Draw
from sklearn.preprocessing import MinMaxScaler
from PIL import Image, ImageDraw, ImageFont

import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import Draw
from sklearn.preprocessing import MinMaxScaler
from PIL import Image, ImageDraw, ImageFont
from rdkit.Chem.Draw import rdMolDraw2D
from io import BytesIO
from PIL import Image, ImageOps

def _drawerToImage(d2d):
    sio = BytesIO(d2d.GetDrawingText())
    return Image.open(sio)

def clourMol(mol, highlightAtoms_p=None, highlightAtomColors_p=None, highlightBonds_p=None, highlightBondColors_p=None, bondWidths_p=None, sz=[400, 400]):
    '''绘制分子图，支持高亮显示原子和键及其颜色，支持不同的边粗细和颜色深度'''

    d2d = rdMolDraw2D.MolDraw2DCairo(sz[0], sz[1])
    op = d2d.drawOptions()
    op.dotsPerAngstrom = 20
    op.bondLineWidth = 1.5  # 修改默认的边宽度为 2.0
    op.useBWAtomPalette()  # 使用黑白配色
    mc = rdMolDraw2D.PrepareMolForDrawing(mol)
    
    # 设置默认的边宽度（如果未提供）
    if bondWidths_p is not None:
        for i, bond in enumerate(mol.GetBonds()):
            # 如果提供了bondWidths_p，则按照对应的宽度设置每个键
            bond.SetProp('mol2d_bondWidth', str(bondWidths_p[i]))

    # 绘制分子图，考虑键的宽度和颜色
    d2d.DrawMolecule(mc, legend='', highlightAtoms=highlightAtoms_p, 
                     highlightAtomColors=highlightAtomColors_p, 
                     highlightBonds=highlightBonds_p, 
                     highlightBondColors=highlightBondColors_p)
    
    d2d.FinishDrawing()
    product_img = _drawerToImage(d2d)
    return product_img

def StripAlphaFromImage(img):
    '''去除图片中的透明通道，返回RGB图片'''
    if len(img.split()) == 3:
        return img
    return Image.merge('RGB', img.split()[:3])

def TrimImgByWhite(img, padding=10):
    '''裁剪图片，去掉周围的白色空白部分'''
    as_array = np.array(img)  # 将图片转换为数组
    if as_array.shape[2] == 4:
        as_array[as_array[:, :, 3] == 0] = [255, 255, 255, 0]

    as_array = as_array[:, :, :3]

    # 寻找非白色区域
    has_content = np.sum(as_array, axis=2, dtype=np.uint32) != 255 * 3
    xs, ys = np.nonzero(has_content)

    # 裁剪区域并加上边框
    margin = 5
    x_range = max([min(xs) - margin, 0]), min([max(xs) + margin, as_array.shape[0]])
    y_range = max([min(ys) - margin, 0]), min([max(ys) + margin, as_array.shape[1]])
    as_array_cropped = as_array[x_range[0]:x_range[1], y_range[0]:y_range[1], 0:3]

    img = Image.fromarray(as_array_cropped, mode='RGB')
    return ImageOps.expand(img, border=padding, fill=(255, 255, 255))


import matplotlib.pyplot as plt
import numpy as np
from rdkit import Chem
from rdkit.Chem import Draw
from sklearn.preprocessing import MinMaxScaler

def visualize_attention_molecules(attention_scores, bg, save_dir=None, atom_type_map=None, threshold=0):
    """
    根据注意力矩阵和分子图批次可视化每个分子，并根据注意力分数渲染颜色。
    注意力矩阵形状为: [batch_size, num_attention_heads, num_nodes, num_nodes]
    
    Parameters:
    - attention_scores: 注意力矩阵，形状为 [batch_size, num_attention_heads, num_nodes, num_nodes]
    - bg: 批次中的图
    - save_dir: 可选，保存图像的目录
    - atom_type_map: 一个字典，将原子类型映射到序号（整数）
    - threshold: 小于此阈值的边设置为浅灰色，大于此阈值的边按注意力大小设置深度
    """
    
    if attention_scores is None:
        raise ValueError("The attention scores are None. Please check the model output.")

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
    
    attention_scores = attention_scores[-1]  # 选择最后一层的注意力分数
    for batch_idx, g in enumerate(gs):
        # 获取图的注意力矩阵（选择第一个注意力头）
        attention_matrix = attention_scores[batch_idx][0].detach().cpu().numpy()

        # 获取图的节点数
        num_nodes = g.num_nodes()

        # 获取边列表
        src_nodes, dst_nodes = g.edges()

        # 提取注意力矩阵的有效部分
        attention_matrix = attention_matrix[:num_nodes, :num_nodes]

        # 获取边上的注意力值
        edge_attention_values = []
        for u, v in zip(src_nodes, dst_nodes):
            if u.item() < num_nodes and v.item() < num_nodes:
                edge_attention_values.append(attention_matrix[u.item(), v.item()])
            else:
                edge_attention_values.append(0)  # 若索引无效，设置为0

        if len(edge_attention_values) == 0:
            print(f"Warning: No valid edge attention values for batch {batch_idx}. Skipping normalization.")
            continue  # 跳过该图

        # 归一化注意力值到 [0, 1]
        scaler = MinMaxScaler()
        edge_attention_values = np.array(edge_attention_values).reshape(-1, 1)
        edge_attention_values_normalized = scaler.fit_transform(edge_attention_values).flatten()

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
        smiles = Chem.MolToSmiles(mol)

        # 使用新的图像大小
        sz = [500, 500]

        # 颜色设置：小于阈值的边为浅灰色，大于阈值的边为绿色，深度根据注意力值调整
        green_color_map = []
        for attention in edge_attention_values_normalized:
            if attention < threshold:  # 小于阈值的边设置为浅灰色
                green_color_map.append((0.8, 0.8, 0.8))  # 浅灰色
            else:  # 大于阈值的边根据注意力值调整绿色深度
                green_color_map.append((0, float(1- attention), 0))  # 从浅绿到深绿
        
        # 根据注意力值调整边的宽度
        bond_widths = np.clip(edge_attention_values_normalized * 4, 1, 6)  # 根据注意力值调整边的宽度

        # `highlightBonds_p` 为列表，包含所有边的索引
        highlight_bonds_list = list(range(len(added_bonds)))

        # 将 `highlightBondColors_p` 转换为字典，边的索引对应颜色
        highlight_bonds_dict = {i: green_color_map[i] for i in range(len(added_bonds))}
        
        # 将 `bondWidths_p` 转换为字典，边的索引对应宽度
        bond_widths_dict = {i: bond_widths[i] for i in range(len(added_bonds))}
        
        # 绘制分子结构图像
        img = clourMol(mol, highlightAtoms_p={},  # 空字典，表示不高亮任何原子
                       highlightAtomColors_p={},  # 空字典，表示不高亮任何原子颜色
                       highlightBonds_p=highlight_bonds_list,  # 边的索引列表
                       highlightBondColors_p=highlight_bonds_dict,  # 使用边的颜色字典
                       bondWidths_p=bond_widths_dict,  # 使用边宽度字典
                       sz=sz)

        # 显示或保存图像
        if save_dir:
            img.save(f"{save_dir}/molecule_{batch_idx}.png")
            print(f"Graph {batch_idx} with SMILES: {smiles} saved to {save_dir}/molecule_{batch_idx}.png")
        else:
            img.show()


def compute_shortest_path_matrix(g):
    """
    计算DGL图 g 的最短路径矩阵
    
    参数：
    g: DGLGraph
        输入的DGL图
    
    返回：
    shortest_path_matrix: np.array
        节点间最短路径矩阵
    """
    # 将图迁移到CPU以便转换为NetworkX图
    g_cpu = g.cpu()

    # 将 DGL 图转换为 NetworkX 图
    nx_graph = g_cpu.to_networkx().to_undirected()

    # 计算最短路径长度（使用 NetworkX 内置的最短路径算法）
    shortest_path_lengths = dict(nx.all_pairs_shortest_path_length(nx_graph))

    num_nodes = g.num_nodes()
    shortest_path_matrix = np.zeros((num_nodes, num_nodes))

    for src, lengths in shortest_path_lengths.items():
        for dst, length in lengths.items():
            shortest_path_matrix[src, dst] = length

    return shortest_path_matrix

def pad_matrix(matrix, max_size):
    """
    对最短路径矩阵进行补零，使其扩展为 (max_size, max_size) 大小。
    
    参数：
    matrix: np.array
        需要补零的最短路径矩阵
    max_size: int
        目标矩阵的大小
    
    返回：
    padded_matrix: np.array
        补零后的矩阵
    """
    padded_matrix = np.zeros((max_size, max_size))
    original_size = matrix.shape[0]
    padded_matrix[:original_size, :original_size] = matrix
    return padded_matrix

def batch_shortest_path_matrices(bg):
    """
    对一个batch的DGL图，计算它们的最短路径矩阵，并补零拼接。
    
    参数：
    gs: list of DGLGraph
        一批DGL图
    
    返回：
    batch_padded_matrices: torch.Tensor
        补零并拼接后的最短路径矩阵
    """
    sg = bg.remove_self_loop()
    gs = dgl.unbatch(sg)
    # 计算最短路径矩阵
    shortest_path_matrices = [compute_shortest_path_matrix(g) for g in gs]
    
    # 找出 batch 中图的最大节点数
    max_size = max([mat.shape[0] for mat in shortest_path_matrices])
    
    # 对每个最短路径矩阵进行补零
    padded_matrices = [pad_matrix(mat, max_size) for mat in shortest_path_matrices]
    
    # 将列表转换为 numpy 数组
    padded_matrices_np = np.array(padded_matrices)
    
    # 然后将 numpy 数组转换为 torch 张量
    return torch.tensor(padded_matrices_np, dtype=torch.float32)


def pair_atom_feats(g, node_feats):
    sg = g.remove_self_loop() # in case g includes self-loop
    atom_idx1, atom_idx2 = sg.edges()
    atom_pair_feats = torch.cat((node_feats[atom_idx1.long()], node_feats[atom_idx2.long()]), dim = 1)
    return atom_pair_feats

def unbatch_mask(bg, atom_feats, bond_feats):

    edit_feats = []
    masks = []
    #num_atoms = []
    feat_dim = atom_feats.size(-1)
    sg = bg.remove_self_loop()
    sg.ndata['h'] = atom_feats
    sg.edata['e'] = bond_feats
    gs = dgl.unbatch(sg)
    
    for idx, g in enumerate(gs):
        e_feats =  g.ndata['h'] #torch.cat((g.ndata['h'], g.edata['e']), dim = 0)  
        mask = torch.ones(e_feats.size()[0], dtype=torch.uint8)
        #num_atom = g.num_nodes()
        edit_feats.append(e_feats)
        masks.append(mask)

    edit_feats = pad_sequence(edit_feats, batch_first=True, padding_value= 0)
    masks = pad_sequence(masks, batch_first=True, padding_value= 0)
    
    return edit_feats, masks

import dgl
import torch

def extend_batch_graph(bg):
    """
    将一个批次图中的每个子图扩展到指定的最大节点数，新增节点保持孤立。

    参数:
    - bg: 输入批次图 (DGLBatch)
    - max_num_nodes: 扩展后的目标节点数量

    返回:
    - new_batch_graph: 扩展后的批次图 (DGLBatch)
    """
    # 将批次图拆分为单个图
    graphs = dgl.unbatch(bg)
    expanded_graphs = []
    max_num_nodes = max(g.num_nodes() for g in graphs)  # 确定批次中最大的节点数

    # 遍历每个子图，进行节点扩展
    for g in graphs:
        # 获取当前图的节点特征
        num_nodes = g.num_nodes()
        
        # 检查是否需要扩展
        if num_nodes < max_num_nodes:
            # 创建新图并扩展节点数量
            new_g = g.clone()
            
            # 计算需要添加的孤立节点数量
            padding_size = max_num_nodes - num_nodes
            
            # 生成零向量特征作为新节点的特征
            feat_dim = g.ndata['h'].size(-1) if 'h' in g.ndata else 0
            if feat_dim > 0:
                padding_feats = torch.zeros((padding_size, feat_dim), device=g.device)
                new_g.add_nodes(padding_size, data={'h': padding_feats})
            else:
                # 如果没有节点特征，只扩展节点数
                new_g.add_nodes(padding_size)
        else:
            new_g = g  # 如果节点数量已达到最大节点数，直接使用原图
        
        # 将扩展后的图添加到列表
        expanded_graphs.append(new_g)

    # 将扩展后的图重新合并为一个批次图
    new_batch_graph = dgl.batch(expanded_graphs)
    
    return new_batch_graph


def compute_node_degrees(bg):
    sg = bg.remove_self_loop()
    gs = dgl.unbatch(sg)
    node_degrees = []

    for g in gs:
        degrees = g.in_degrees().float()
        node_degrees.append(degrees)

    # 拼接所有节点的度数为一个一维张量
    all_degrees = torch.cat(node_degrees)

    return all_degrees


def get_bond_feats_matrices(bg, bond_feats):
    """
    输入：batch_graph，一个批处理的DGL图
    输出：一个包含每个子图的边特征矩阵的列表
          每个矩阵的大小为：max_num_atoms x max_num_atoms x 边特征维度
          如果两个原子之间没有边，则相应矩阵元素填充为0
    """
    # 尝试移除自环
    sg = bg.remove_self_loop()
    
    sg.edata['e'] = bond_feats
    gs = dgl.unbatch(sg)
    
    # 获取批量中的最大节点数
    max_num_nodes = max(g.num_nodes() for g in gs)
    
    # 边特征维度 (假设每条边的特征是相同的维度)
    edge_feat_dim = sg.edata['e'].shape[1] 

    # 获取batch中的分子图数量
    batch_size = len(gs)
    
     # 初始化存储所有分子图边特征的tensor, 形状为: batch_size x max_num_nodes x max_num_nodes x edge_feat_dim
    bond_feature_matrices = torch.zeros((batch_size, max_num_nodes, max_num_nodes, edge_feat_dim), dtype=torch.float32)
    
    # 将batch分解为单独的分子图
    for idx, g in enumerate(gs):
        num_nodes = g.num_nodes()
        
        # 获取每条边的索引 (起始节点和结束节点) 以及它们的边特征
        src, dst = g.edges()
        edge_features = g.edata['e']
        
        # 将边特征填充到对应的矩阵位置
        for i in range(len(src)):
            bond_feature_matrices[idx, src[i], dst[i]] = edge_features[i]
    
        return bond_feature_matrices

def unbatch_feats(bg, edit_feats):
    sg = bg.remove_self_loop()
    gs = dgl.unbatch(sg)
    atom_feats = []
    bond_feats = []
    for i, g in enumerate(gs):
        atom_feats.append(edit_feats[i][:g.num_nodes()])
        bond_feats.append(edit_feats[i][g.num_nodes():g.num_nodes()+g.num_edges()])
    return torch.cat(atom_feats, dim = 0)#, torch.cat(bond_feats, dim = 0)

class Global_Reactivity_Attention(nn.Module):
    def __init__(self, d_model, heads, n_layers = 1, dropout = 0.1):
        super(Global_Reactivity_Attention, self).__init__()
        self.n_layers = n_layers
        att_stack = []
        pff_stack = []
        for _ in range(n_layers):
            att_stack.append(MultiHeadAttention(heads, d_model, dropout))
            pff_stack.append(FeedForward(d_model, dropout))
        self.att_stack = nn.ModuleList(att_stack)
        self.pff_stack = nn.ModuleList(pff_stack)
        
    def forward(self, x, mask, attn_bias=None):
        scores = []
        for n in range(self.n_layers):
            score, x = self.att_stack[n](x, mask, attn_bias)
            x = self.pff_stack[n](x)
            scores.append(score)
        return scores, x

class Global_Reactivity_Attention_GNN(nn.Module):
    def __init__(self, d_model, heads, batch_norm, residual, n_layers = 1, dropout = 0.1,  device="cpu"):
        super(Global_Reactivity_Attention_GNN, self).__init__()
        self.n_layers = n_layers
        self.gcn_layers = nn.ModuleList([GatedGCNLayer(d_model, d_model, dropout,
                                                    batch_norm, residual)  for _ in range(n_layers // 2)])
        self.att_layers = nn.ModuleList([MultiHeadAttention(heads, d_model, dropout) for _ in range(n_layers // 2)])
        self.pff_stack = nn.ModuleList([FeedForward(d_model, dropout) for _ in range(n_layers)])
        
    def forward(self, g, x, e, mask, attn_bias=None):
        scores = []
        gcn_index = 0
        att_index = 0
        batch_size, num_nodes, feature_dim = x.size()
        for n in range(self.n_layers):
            if n % 2 == 0:
                # 偶数层使用 GCN 聚合局部特征
                # 将 x 转换为 (batch_size * num_nodes, feature_dim) 以满足 GCN 层需求
                x_gcn = x.view(batch_size * num_nodes, feature_dim)
                x, e = self.gcn_layers[gcn_index](g, x_gcn, e)
                # 将 GCN 输出再 reshape 回 (batch_size, num_nodes, feature_dim)
                x = x.view(batch_size, num_nodes, feature_dim)
                gcn_index += 1
            else:
                # 奇数层使用全局注意力
                score, x = self.att_layers[att_index](x, mask, attn_bias)
                scores.append(score)
                att_index += 1
            
            # 每层后进行 MoE 操作
            x = self.pff_stack[n](x)
        return scores, x

class Global_Reactivity_Attention_MoE_GNN(nn.Module):
    def __init__(self, d_model, heads, batch_norm, residual, n_layers = 1, dropout = 0.1, num_experts=8, num_experts_per_tok=2,  device="cpu"):
        super(Global_Reactivity_Attention_MoE_GNN, self).__init__()
        self.n_layers = n_layers
        self.gcn_layers = nn.ModuleList([GatedGCNLayer(d_model, d_model, dropout,
                                                    batch_norm, residual)  for _ in range(n_layers // 2)])
        self.att_layers = nn.ModuleList([MultiHeadAttention(heads, d_model, dropout) for _ in range(n_layers // 2)])
        self.pff_stack = nn.ModuleList([MoE(num_experts=num_experts, num_experts_per_tok=num_experts_per_tok, 
                                            d_model=d_model, device=device, dropout=dropout) for _ in range(n_layers)])
        
    def forward(self, g, x, e, mask, attn_bias=None):
        scores = []
        gcn_index = 0
        att_index = 0
        batch_size, num_nodes, feature_dim = x.size()
        for n in range(self.n_layers):
            if n % 2 == 0:
                # 偶数层使用 GCN 聚合局部特征
                # 将 x 转换为 (batch_size * num_nodes, feature_dim) 以满足 GCN 层需求
                x_gcn = x.view(batch_size * num_nodes, feature_dim)
                x, e = self.gcn_layers[gcn_index](g, x_gcn, e)
                # 将 GCN 输出再 reshape 回 (batch_size, num_nodes, feature_dim)
                x = x.view(batch_size, num_nodes, feature_dim)
                gcn_index += 1
            else:
                # 奇数层使用全局注意力
                score, x = self.att_layers[att_index](x, mask, attn_bias)
                scores.append(score)
                att_index += 1
            
            # 每层后进行 MoE 操作
            x = self.pff_stack[n](x)
        return scores, x, e


class Global_Reactivity_Attention_MoE(nn.Module):
    def __init__(self, d_model, heads, n_layers = 1, dropout = 0.1, num_experts=8, num_experts_per_tok=2,  device="cpu"):
        super(Global_Reactivity_Attention_MoE, self).__init__()
        self.n_layers = n_layers
        att_stack = []
        pff_stack = []
        for _ in range(n_layers):
            att_stack.append(MultiHeadAttention(heads, d_model, dropout))
            pff_stack.append(MoE(num_experts=num_experts, num_experts_per_tok=num_experts_per_tok, 
                                    d_model=d_model, device=device, dropout=dropout))
        self.att_stack = nn.ModuleList(att_stack)
        self.pff_stack = nn.ModuleList(pff_stack)
        
    def forward(self, x, mask, attn_bias=None):
        scores = []
        for n in range(self.n_layers):
            score, x = self.att_stack[n](x, mask, attn_bias)
            x = self.pff_stack[n](x)
            scores.append(score)
        return scores, x


class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model, dropout = 0.1):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads
        self.q_linear = nn.Linear(d_model, d_model, bias=False)
        self.v_linear = nn.Linear(d_model, d_model, bias=False)
        self.k_linear = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.reset_parameters()
        
    def reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
                
    def attention(self, q, k, v, mask=None, attn_bias=None):  ##############################新增attn_bias: Tensor = None,

        scores = torch.matmul(q, k.transpose(-2, -1)) /  math.sqrt(self.d_k)

        if attn_bias is not None:
            scores += attn_bias
        if mask is not None:

            #mask = mask.to(scores.device)
            #zero_out_last_n_rows = (mask == 0).sum(dim=1)  ##新加 修改后几行不为0的bug

            mask = mask.unsqueeze(1).repeat(1,mask.size(-1),1)                    
            mask = mask.unsqueeze(1).repeat(1,scores.size(1),1,1)

            """
            # 对每个样本，应用零掩码  
            for i in range(scores.size(0)):      ##新加 修改后几行不为0的bug
                if zero_out_last_n_rows[i] > 0:
                    mask[i, :, -zero_out_last_n_rows[i]:, :] = 0
            #scores = scores.masked_fill(mask == 0, float('-inf'))
            """
            scores[~mask.bool()] = float('-9e15')  #-9e15
        scores = torch.softmax(scores, dim=-1)

        """
        if mask is not None:
            scores = scores.clone()
            # 对每个样本，应用零掩码  
            for i in range(scores.size(0)):      ##新加 修改后几行不为0的bug
                if zero_out_last_n_rows[i] > 0:
                    scores[i, :, -zero_out_last_n_rows[i]:, :] = 0
        """
        scores = self.dropout(scores)
        output = torch.matmul(scores, v)
        return scores, output

    def forward(self, x, mask=None, attn_bias=None):    ##############################新增 attn_bias: Tensor = None,
        bs = x.size(0)
        k = self.k_linear(x).view(bs, -1, self.h, self.d_k)
        q = self.q_linear(x).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(x).view(bs, -1, self.h, self.d_k)
        k = k.transpose(1,2)
        q = q.transpose(1,2)
        v = v.transpose(1,2)
        scores, output = self.attention(q, k, v, mask, attn_bias)  ##############################新增attn_bias,
        output = output.transpose(1,2).contiguous().view(bs, -1, self.d_model)
        output = output + x
        output = self.layer_norm(output)
        return scores, output.squeeze(-1)

class GELU(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2/math.pi) * (x + 0.044715 * torch.pow(x, 3)))) 
       
class FeedForward(nn.Module):
    def __init__(self, d_model, dropout = 0.1):
        super(FeedForward, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model*2),
            GELU(),
            nn.Linear(d_model*2, d_model),
            nn.Dropout(dropout)
        )
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        
    def forward(self, x):
        output = self.net(x)
        return self.layer_norm(x + output)


class MoE(nn.Module):
    def __init__(self, num_experts, num_experts_per_tok, d_model, device, dropout=0.1):
        super().__init__()
        # 初始化专家
        self.experts = nn.ModuleList([FeedForward(d_model,dropout).to(device) for i in range(num_experts)])
        # 门控线性层
        self.gate = nn.Linear(d_model, num_experts, bias=False)
        # 路由的专家数量
        self.num_experts_per_tok = num_experts_per_tok

    def forward(self, x):
        orig_shape = x.shape
        # (b, n, d) -> (b*n, d)
        x = x.view(-1, x.shape[-1])
		
        # shape: (b*n, num_experts)
        scores = self.gate(x)
        # (b*n, k), 一个是权重，一个是索引
        expert_weights, expert_indices = torch.topk(scores, self.num_experts_per_tok, dim=-1)
        # 归一化，k个权重和为1
        expert_weights = expert_weights.softmax(dim=-1)
        # (b*n, k) -> (b*n*k,)
        flat_expert_indices = expert_indices.view(-1)
		
        # (b*n, d) -> (b*n*k, d)
        x = x.repeat_interleave(self.num_experts_per_tok, dim=0)
        y = torch.empty_like(x)
        for i, expert in enumerate(self.experts):
            # 根据索引进行路由，将每个token输入索引对应的专家
            y[flat_expert_indices == i] = expert(x[flat_expert_indices == i])
        # (b*n, k, d) * (b*n, k, 1) -> (b*n, d)
        y = (y.view(*expert_weights.shape, -1) * expert_weights.unsqueeze(-1)).sum(dim=1)
        return y.view(*orig_shape)  # (b*n, d) -> (b, n, d)


class GatingNetwork(nn.Module):
    def __init__(self, input_dim):
        super(GatingNetwork, self).__init__()
        self.fc = nn.Linear(input_dim * 2, 1)  # GNN和Transformer的特征拼接后的输入
        self.sigmoid = nn.Sigmoid()

    def forward(self, gnn_out, transformer_out):
        # 拼接GNN和Transformer的输出
        combined = torch.cat([gnn_out, transformer_out], dim=-1)
        gate = self.fc(combined)
        gate = self.sigmoid(gate)
        return gate

def visualize_attention_matrices(attention_score, save_dir):
    # 确保保存目录存在
    os.makedirs(save_dir, exist_ok=True)
    
    # 提取第一个头部的注意力矩阵
    attention_score = attention_score[0]
    first_head_attention = attention_score[:, 0, :, :]
    
    # 遍历每个样本并生成注意力矩阵图像
    for sample_index in range(first_head_attention.shape[0]):
        attention_matrix = first_head_attention[sample_index].detach().cpu().numpy()
        
        # 可视化并保存第一个头部的注意力矩阵
        plt.figure(figsize=(10, 8))
        plt.imshow(attention_matrix, cmap='viridis')
        plt.colorbar()
        plt.title(f'Attention Matrix for Sample {sample_index} (Head 1)')
        plt.xlabel('Key')
        plt.ylabel('Query')
        
        # 保存图像
        save_path = os.path.join(save_dir, f'attention_matrix_sample_{sample_index}_head_1.png')
        plt.savefig(save_path)
        plt.close()  # 关闭图形以释放内存

    print(f"Saved all attention matrices in {save_dir}")


class MLPReadout(nn.Module):

    def __init__(self, input_dim, output_dim, L=2): #L=nb_hidden_layers
        super().__init__()
        list_FC_layers = [ nn.Linear( input_dim//2**l , input_dim//2**(l+1) , bias=True ) for l in range(L) ]
        list_FC_layers.append(nn.Linear( input_dim//2**L , output_dim , bias=True ))
        self.FC_layers = nn.ModuleList(list_FC_layers)
        self.L = L
        
    def forward(self, x):
        y = x
        for l in range(self.L):
            y = self.FC_layers[l](y)
            #y = F.relu(y)
            y = F.gelu(y)
            nn.Dropout(0.2),
        y = self.FC_layers[self.L](y)
        return y

# 分类器部分
class Classifier1(nn.Module):
    def __init__(self, input_dim, output_dim, num_tasks):
        super(Classifier1, self).__init__()
        self.fc = MLPReadout(input_dim, output_dim)

    def forward(self, x):
        return self.fc(x)

class Classifier2(nn.Module):
    def __init__(self, input_dim, output_dim, num_tasks):
        super(Classifier2, self).__init__()
        self.fc = MLPReadout(input_dim, output_dim)

    def forward(self, x):
        return self.fc(x)

class Classifier3(nn.Module):
    def __init__(self, input_dim, output_dim, num_tasks):
        super(Classifier3, self).__init__()
        self.fc = MLPReadout(input_dim, output_dim)

    def forward(self, x):
        return self.fc(x)
    

