import matplotlib.pyplot as plt
import networkx as nx
import torch
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def visualize_attention_matrices(attention_scores, g, save_dir=None):
    """
    根据注意力矩阵和分子图可视化分子，并根据注意力分数渲染颜色。
    """
    # 取最后一层的注意力矩阵
    attention_matrix = attention_scores[-1].detach().cpu().numpy()  # 最后一层的注意力矩阵
    print(f"Attention matrix shape: {attention_matrix.shape}")

    # 获取分子图的节点和边
    edge_list = g.edges()
    edge_attention_values = []

    # 假设每条边的注意力矩阵中的对应元素是该边的注意力值
    for i, (u, v) in enumerate(edge_list):
        # 注意力矩阵的每一行/列对应分子中的节点，这里假设是按顺序的
        edge_attention_values.append(attention_matrix[0, u.item(), v.item()])

    # 归一化注意力值到 [0, 1]
    scaler = MinMaxScaler()
    edge_attention_values = np.array(edge_attention_values).reshape(-1, 1)
    edge_attention_values_normalized = scaler.fit_transform(edge_attention_values).flatten()

    # 创建一个NetworkX图
    G = nx.Graph()

    # 添加节点
    for i in range(g.num_nodes()):
        G.add_node(i)

    # 添加边并设置颜色
    edge_colors = []
    for i, (u, v) in enumerate(edge_list):
        edge_colors.append(plt.cm.viridis(edge_attention_values_normalized[i]))

    # 绘制图形
    pos = nx.spring_layout(G)  # 使用 spring_layout 来进行节点布局
    plt.figure(figsize=(8, 6))
    nx.draw(G, pos, node_size=500, node_color='lightblue', with_labels=True, font_weight='bold', font_size=10,
            edge_color=edge_colors, width=3, alpha=0.7)
    plt.title('Molecular Graph with Attention Visualization')

    if save_dir:
        plt.savefig(save_dir)
        print(f"Graph saved to {save_dir}")
    else:
        plt.show()

