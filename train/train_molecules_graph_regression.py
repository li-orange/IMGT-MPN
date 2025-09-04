"""
    Utility functions for training one epoch 
    and evaluating one epoch
"""
import torch

from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import MinMaxScaler

import dgl
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

def evaluate_network_sparse_new(model, device, data_loader, num_tasks, task_type, dataset_idx, test_val_flag = None):
    model.eval()
    epoch_test_loss = 0
    epoch_test_mae = 0
    epoch_test_rmse = 0  # 使用 RMSE
    nb_data = 0
    #计算auc
    average_auc = 0
    batch_idx = 0
    targets = []
    scores = []
    
    with torch.no_grad():
        for batch_idx, (batch_graphs, batch_targets, _) in enumerate(data_loader):
            batch_graphs = batch_graphs.remove_self_loop()  ###如果不用图卷积就去掉自环
            batch_graphs = batch_graphs.to(device)
            batch_x = batch_graphs.ndata['feat'].to(device)
            batch_e = batch_graphs.edata['feat'].to(device)
            
            num_task = num_tasks[dataset_idx]
            #调整维度
            batch_targets = batch_targets.squeeze()
            
            batch_targets = batch_targets.to(device)

            training_flag = False
            try:
                batch_pos_enc = batch_graphs.ndata['pos_enc'].to(device)
                batch_scores,attention_score = model.forward(dataset_idx, batch_graphs, batch_x, batch_e, training_flag, batch_pos_enc)
            except:
                batch_scores,attention_score = model.forward(dataset_idx, batch_graphs, batch_x, batch_e, training_flag)
            #调整维度
            batch_scores = batch_scores.squeeze() 
            loss = model.loss(batch_scores, batch_targets, num_task)

            #visualize_attention_molecules(attention_score, batch_graphs, save_dir="attention_visualization")

            if batch_targets.ndimension() == 0:  # 如果是标量
                batch_targets = batch_targets.unsqueeze(0)  # 将其转换为一维数组
            if batch_scores.ndimension() == 0:  # 如果是标量
                batch_scores = batch_scores.unsqueeze(0)  # 将其转换为一维数组
            # 根据任务类型选择损失函数
            if task_type == 'reg':
                # 回归任务
                targets.extend(batch_targets.detach().cpu().numpy()) 
                scores.extend(batch_scores.detach().cpu().numpy()) 
                # 累积 RMSE
                epoch_test_rmse += loss.item()* batch_targets.size(0)  # 累积 RMSE
            elif task_type == 'cls' and num_task == 1:
                # 单标签分类任务，可能需要 one-hot 编码
                scatter_targets = torch.zeros(batch_targets.size(0), batch_scores.size(1)).to(device)
                scatter_targets.scatter_(1, batch_targets.view(-1, 1), 1)
                targets.extend(scatter_targets.detach().cpu().numpy()) 
                scores.extend(batch_scores.detach().cpu().numpy()) 
            elif task_type == 'cls' and num_task > 1:
                # 多标签分类任务，不需要 one-hot 编码
                targets.extend(batch_targets.detach().cpu().numpy()) 
                scores.extend(batch_scores.detach().cpu().numpy()) 

            epoch_test_loss += loss.detach().item()
            #epoch_test_mae += MAE(batch_scores, batch_targets)
            nb_data += batch_targets.size(0)
        
        epoch_test_loss /= (batch_idx + 1)
        epoch_test_rmse /= nb_data  # 计算平均 RMSE

        if task_type == 'cls':
            #计算auc
            average_auc = roc_auc_score(targets, scores)
            if test_val_flag == 'test':
                print("test_AUC:",average_auc)
            else:
                print("val_AUC:",average_auc)
                
        elif task_type == 'reg': 
            # 输出回归任务的 RMSE
            if test_val_flag == 'test':
                print("test_RMSE:",epoch_test_rmse)
            else:
                print("val_RMSE:",epoch_test_rmse)
        
    return epoch_test_loss, epoch_test_rmse, average_auc


def train_epoch_sparse_interleaved(model, optimizer, device, interleaved_loader, num_tasks, task_type):
    """
    每个批次交替训练多个数据集。
    参数:
        model: 模型对象
        optimizer: 优化器
        device: 设备（CPU 或 GPU）
        interleaved_loader: 交替加载器
        num_tasks: 任务数量
        task_type: 任务类型（'cls' 或 'reg'）
    返回:
        当前 epoch 的损失、优化器、以及（分类任务的）AUC
    """
    model.train()
    epoch_loss = 0
    epoch_train_mae = 0
    epoch_train_rmse = 0  # 使用 RMSE
    nb_data = 0
    gpu_mem = 0
    #计算auc
    average_auc = 0
    targets = []
    scores = []
    batch_idx = 0
    attention_score = []
    batch_graphs = []

    for batch_idx, (batch_data, idx) in enumerate(interleaved_loader):
        batch_graphs, batch_targets, dataset_idx = batch_data

        num_task = num_tasks[dataset_idx]

        # 检查 batch 是否为空，避免无效计算
        if batch_graphs.batch_size == 0 or len(batch_targets) == 0:

            print(f"跳过空批次，数据集索引: {dataset_idx}, 批次索引: {batch_idx}")
            continue
        
        batch_graphs = batch_graphs.remove_self_loop()  ###如果不用图卷积就去掉自环
        batch_graphs = batch_graphs.to(device)
        batch_x = batch_graphs.ndata['feat'].to(device)  # num x feat
        batch_e = batch_graphs.edata['feat'].to(device)
        #调整维度
        batch_targets = batch_targets.squeeze()
        
        batch_targets = batch_targets.to(device)
        optimizer.zero_grad()
        
        training_flag = True
        try:
            batch_pos_enc = batch_graphs.ndata['pos_enc'].to(device)
            sign_flip = torch.rand(batch_pos_enc.size(1)).to(device)
            sign_flip[sign_flip>=0.5] = 1.0; sign_flip[sign_flip<0.5] = -1.0
            batch_pos_enc = batch_pos_enc * sign_flip.unsqueeze(0)
            batch_scores,attention_score = model.forward(dataset_idx, batch_graphs, batch_x, batch_e, training_flag, batch_pos_enc)
        except:
            batch_scores,attention_score = model.forward(dataset_idx, batch_graphs, batch_x, batch_e, training_flag)
        
        #调整维度
        batch_scores = batch_scores.squeeze()

        loss = model.loss(batch_scores, batch_targets, num_task)
        
        if batch_targets.ndimension() == 0:  # 如果是标量
            batch_targets = batch_targets.unsqueeze(0)  # 将其转换为一维数组
        if batch_scores.ndimension() == 0:  # 如果是标量
            batch_scores = batch_scores.unsqueeze(0)  # 将其转换为一维数组

        # 根据任务类型选择损失函数
        if task_type == 'reg':
            # 回归任务
            targets.extend(batch_targets.detach().cpu().numpy()) 
            scores.extend(batch_scores.detach().cpu().numpy()) 
            # 累积 RMSE
            epoch_train_rmse += loss.item()* batch_targets.size(0)  # 累积 RMSE
        elif task_type == 'cls' and num_task == 1:
            # 单标签分类任务，可能需要 one-hot 编码
            scatter_targets = torch.zeros(batch_targets.size(0), batch_scores.size(1)).to(device)
            scatter_targets.scatter_(1, batch_targets.view(-1, 1), 1)
            targets.extend(scatter_targets.detach().cpu().numpy()) 
            scores.extend(batch_scores.detach().cpu().numpy()) 
        elif task_type == 'cls' and num_task > 1:
            # 多标签分类任务，不需要 one-hot 编码
            targets.extend(batch_targets.detach().cpu().numpy()) 
            scores.extend(batch_scores.detach().cpu().numpy()) 
        
        loss.backward()
        optimizer.step()
        epoch_loss += loss.detach().item()
        #epoch_train_mae += MAE(batch_scores, batch_targets)
        nb_data += batch_targets.size(0)

    epoch_loss /= (batch_idx + 1)
    epoch_train_rmse /= nb_data  # 计算平均 RMSE
    

    average_auc = 0.0
    if task_type == 'cls' and len(targets) > 0 and len(scores) > 0:
        try:
            average_auc = roc_auc_score(targets, scores)
            print(f"Train AUC: {average_auc:.4f}")
        except ValueError as e:
            print(f"AUC 计算失败: {e}")
    elif task_type == 'reg':
        # 输出回归任务的 RMSE
        print(f"Train RMSE: {epoch_train_rmse:.4f}")

    return epoch_loss, epoch_train_rmse, optimizer, average_auc



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
        'H', 'C', 'N', 'O', 'F', 'P', 'S', 'Cl', 'Br', 'I', 
        'Ca', 'Na', 'B', 'Mn', 'Zn', 'Co', 'Fe', 'Tl', 'Hg', 'Bi', 
        'Se', 'Cu', 'Au', 'Pt', 'Al', 'Ti', 'Tc', 'Si', 'As', 'Cr', 
        '*'  # Unknown atom type
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

        # 提取SMILES表示
        smiles = dgl_graph_to_smiles(g, atom_type_map)
        
        # 绘制分子结构图
        draw_molecule_from_smiles(smiles, save_path=f"{save_dir}/molecule_{batch_idx}.png" if save_dir else None)

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


from rdkit import Chem

def dgl_graph_to_smiles(dgl_graph, atom_type_map):
    """
    将DGL图转换为SMILES字符串
    """
    mol = Chem.RWMol()  # 创建一个可修改的分子

    # 获取节点和边的信息
    atom_types = dgl_graph.ndata['atom_type'].detach().cpu().numpy()  # 获取原子类型序号
    for atom_idx, atom_type in enumerate(atom_types):
        atom_symbol = list(atom_type_map.keys())[list(atom_type_map.values()).index(atom_type)]
        mol.AddAtom(Chem.Atom(atom_symbol))  # 添加原子到分子中

    src_nodes, dst_nodes = dgl_graph.edges()
    
    # 添加边时检查是否已有相同的键
    for u, v in zip(src_nodes, dst_nodes):
        atom_u_idx = u.item()
        atom_v_idx = v.item()
        
        # 检查是否已存在边
        if not mol.GetBondBetweenAtoms(atom_u_idx, atom_v_idx):
            mol.AddBond(atom_u_idx, atom_v_idx, Chem.BondType.SINGLE)  # 默认添加单键

    # 转换为SMILES
    smiles = Chem.MolToSmiles(mol)
    return smiles


from rdkit.Chem import Draw

def draw_molecule_from_smiles(smiles, save_path=None):
    """
    根据 SMILES 字符串绘制分子结构图并保存
    """
    if smiles is None:
        print("Invalid SMILES string provided. Skipping drawing.")
        return

    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            print(f"Failed to create molecule from SMILES: {smiles}. Skipping drawing.")
            return
        
        img = Draw.MolToImage(mol, size=(300, 300))
        
        if save_path:
            img.save(save_path)
        else:
            img.show()
    except Exception as e:
        print(f"Error drawing molecule: {str(e)}")


