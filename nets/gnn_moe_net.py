import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.nn as dglnn
from layers.mlp_readout_layer import MLPReadout
from layers.gat_layer import GATLayer
# ====== 基础模块 ======

class GELU(nn.Module):
    def forward(self, x):
        return F.gelu(x)

class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, n_layers=2, act=nn.ReLU()):
        super().__init__()
        layers = []
        dims = [in_dim] + [hidden_dim]*(n_layers-1) + [out_dim]
        for i in range(len(dims)-2):
            layers += [nn.Linear(dims[i], dims[i+1]), act]
        layers += [nn.Linear(dims[-2], dims[-1])]
        self.net = nn.Sequential(*layers)
    def forward(self, x): return self.net(x)

# ====== GAT / GIN / GatedGCN 三种骨干 ======

class GATBackbone(nn.Module):
    """
    使用 dgl.nn.GATConv。注意：hidden_dim 需能被 num_heads 整除。
    输出统一为 hidden_dim（多头 concat 后再线性投回 hidden_dim）。
    """
    def __init__(self, in_dim, hidden_dim, num_layers, num_heads, dropout, residual=True):
        super().__init__()
        assert hidden_dim % num_heads == 0, "hidden_dim 必须能被 attention_heads 整除"
        self.layers = nn.ModuleList()
        self.batch_norm = True
        self.residual = residual
        h_head = hidden_dim // num_heads

        # 首层
        self.layers.append(GATLayer(in_dim, h_head, num_heads, dropout, self.batch_norm, self.residual))
        
        # 中间层
        for _ in range(num_layers - 2):
            self.layers.append(GATLayer(hidden_dim, h_head, num_heads, dropout, self.batch_norm, self.residual))
        
        # 末层
        self.layers.append(GATLayer(hidden_dim, hidden_dim, 1, dropout, self.batch_norm, self.residual))

    def forward(self, g, h):
        """
        前向传播，处理图和节点特征。
        """
        for i, conv in enumerate(self.layers):
            h = conv(g, h)  # 计算图卷积
        return h  # 通过线性层映射到 hidden_dim


class GINBackbone(nn.Module):
    """
    使用 dgl.nn.GINConv。每层内置一个小 MLP。
    """
    def __init__(self, in_dim, hidden_dim, num_layers, dropout, residual=True):
        super().__init__()
        self.layers = nn.ModuleList()
        self.residual = residual
        # 首层
        self.layers.append(dglnn.GINConv(MLP(in_dim, hidden_dim, hidden_dim, n_layers=2, act=nn.ReLU()), learn_eps=True))
        # 中间层
        for _ in range(num_layers-1):
            self.layers.append(dglnn.GINConv(MLP(hidden_dim, hidden_dim, hidden_dim, n_layers=2, act=nn.ReLU()), learn_eps=True))
        self.layer_norm = nn.LayerNorm(hidden_dim)  # 备用的 LayerNorm
        self.bn = nn.ModuleList([nn.BatchNorm1d(hidden_dim) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, g, h):
        for i, conv in enumerate(self.layers):
            h_res = h
            h = conv(g, h)
            # 使用 BatchNorm 或 LayerNorm，根据 batch size 判断
            if self.training and h.size(0) > 1:  # 如果在训练阶段并且 batch size > 1
                h = self.bn[i](h)  # 使用 BatchNorm1d
            else:
                h = self.layer_norm(h)  # 否则使用 LayerNorm

            h = F.relu(h)
            h = self.dropout(h)
            if self.residual and h_res.shape == h.shape:
                h = h + h_res
        return h


class GatedGCNLayer(nn.Module):
    """
    轻量版 GatedGCN 层（常见实现的简化）。
    论文：Bresson & Laurent, ICML'18 Workshop / Dwivedi et al., ICLR'21。
    """
    def __init__(self, hidden_dim, dropout=0.0):
        super().__init__()
        self.A = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.B = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.C = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.D = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.E = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def message(self, edges):
        e = torch.sigmoid(edges.src['Ah'] + edges.dst['Bh'] + edges.data['e'])
        m = edges.src['Ch'] * e
        return {'m': m, 'gate': e}

    def forward(self, g, h, e):
        with g.local_scope():
            g.ndata['Ah'] = self.A(h)
            g.ndata['Bh'] = self.B(h)
            g.ndata['Ch'] = self.C(h)
            g.edata['e']  = self.E(e)   # 边嵌入参与 gating
            g.update_all(self.message, dgl.function.sum('m', 'm_sum'))
            h_new = self.D(h) + g.ndata['m_sum']   # 残差样式
            h_new = self.bn(h_new)
            h_new = F.relu(h_new)
            h_new = self.dropout(h_new)
            return h_new, g.edata['e']  # 回传边表征（已线性映射）


class GatedGCNBackbone(nn.Module):
    def __init__(self, hidden_dim, num_layers, dropout, residual=True):
        super().__init__()
        self.layers = nn.ModuleList([GatedGCNLayer(hidden_dim, dropout) for _ in range(num_layers)])
        self.residual = residual

    def forward(self, g, h, e):
        for layer in self.layers:
            h_res = h
            h, e = layer(g, h, e)
            if self.residual and h.shape == h_res.shape:
                h = h + h_res
        return h, e

# ====== MoE 头（图表征后的专家混合，多任务共享） ======

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

# ====== 主模型：可切换骨干 + 可选 MoE 头 ======

class GNNWithOptionalMoE(nn.Module):
    """
    兼容你现有的接口：
      - forward(dataset_idx, g, h, e, training_flag, h_pos_enc=None) -> (output, attention_score)
      - self.loss(scores, targets, num_task)
    支持 backbone_type in {'gat', 'gatedgcn', 'gin'} 与 use_moe True/False
    """
    def __init__(self, net_params):
        super().__init__()
        # ---- 必要超参 ----
        in_atom_feat_dim = net_params['in_atom_feat_dim']
        in_bond_feat_dim = net_params['in_bond_feat_dim']
        self.in_atom_feat_dim = in_atom_feat_dim
        self.in_bond_feat_dim = in_bond_feat_dim
        hidden_dim       = net_params['hidden_dim']
        out_dim          = net_params['out_dim']      # 用于 readout 后维度组合
        in_feat_dropout  = net_params.get('in_feat_dropout', 0.0)
        attention_heads  = net_params.get('attention_heads', 4)
        num_layers       = net_params.get('num_layers', 4)
        dropout          = net_params.get('dropout', 0.0)

        self.readout     = net_params.get('readout', 'mean')
        self.residual    = net_params.get('residual', True)
        self.edge_feat   = net_params.get('edge_feat', True)
        self.device      = net_params['device']
        self.pos_enc     = net_params.get('pos_enc', False)

        self.use_moe     = net_params.get('use_moe', False)
        self.task_type   = net_params['task_type']          # 'cls' or 'reg'
        self.num_tasks   = net_params['num_tasks']          # list like [1] or [2,2,1,...]
        self.activation  = GELU()
        self.backbone_type = net_params.get('backbone_type', 'gat')  # 'gat'|'gatedgcn'|'gin'

        # ---- 嵌入层 ----
        if self.pos_enc:
            pos_enc_dim = net_params['pos_enc_dim']
            self.embedding_pos_enc = nn.Linear(pos_enc_dim, hidden_dim)

        self.embedding_h = nn.Linear(in_atom_feat_dim, hidden_dim)
        if self.edge_feat:
            self.embedding_e = nn.Linear(in_bond_feat_dim, hidden_dim)
        else:
            self.embedding_e = nn.Linear(1, hidden_dim)

        self.in_feat_dropout = nn.Dropout(in_feat_dropout)

        # ---- 选择骨干 ----
        if self.backbone_type == 'gat':
            self.backbone = GATBackbone(hidden_dim, hidden_dim, num_layers, attention_heads, dropout, residual=self.residual)
            self.uses_edge = False
        elif self.backbone_type == 'gin':
            self.backbone = GINBackbone(hidden_dim, hidden_dim, num_layers, dropout, residual=self.residual)
            self.uses_edge = False
        elif self.backbone_type == 'gatedgcn':
            self.backbone = GatedGCNBackbone(hidden_dim, num_layers, dropout, residual=self.residual)
            self.uses_edge = True
        else:
            raise ValueError(f"Unknown backbone_type: {self.backbone_type}")

        # ---- Readout ----
        self.readout_mode = self.readout  # 'sum'|'max'|'mean'|'mean+sum' 等

        # ---- 任务头（多任务） ----
        # 你的原实现把 node/edge readout 拼接 -> out_dim*2 再接各任务头
        # 这里我们保持相同接口：
        head_in_dim = out_dim*2

        # 分类头
        if self.task_type == 'cls':
            if len(self.num_tasks) == 1:
                self.classifier_single = MLPReadout(head_in_dim, 2 if self.num_tasks[0]==1 else self.num_tasks[0])
            else:
                self.classifier1 = MLPReadout(head_in_dim, 2)   # BACE
                self.classifier2 = MLPReadout(head_in_dim, 2)   # BBBP
                self.classifier3 = MLPReadout(head_in_dim, 2)   # ClinTox
                self.classifier4 = MLPReadout(head_in_dim, 27)  # SIDER
                # self.classifier5 = MLPReadout(head_in_dim, 12) # Tox21(如需)
        else:
            # 回归头
            self.regressor1 = MLPReadout(head_in_dim, 1)  # FreeSolv
            self.regressor2 = MLPReadout(head_in_dim, 1)  # Lipophilicity
            self.regressor3 = MLPReadout(head_in_dim, 1)  # ESOL

        # ---- MoE 头（可选）：放在“图级拼接向量 -> 任务头”之间 ----
        if self.use_moe:
            # 这里将 head_in_dim 作为 MoE 输入输出维度（MoE 内部是多专家 MLP，再投回同维度）
            self.moe = MoE(num_experts=4, num_experts_per_tok=2, d_model=head_in_dim, 
                           device=self.device, dropout=dropout)
        else:
            self.moe = None

        # ---- 读出时也计算 edge 聚合（与你原文一致） ----
        self._cached_bn = nn.BatchNorm1d(hidden_dim)

    # ---- Readout helpers ----
    def _graph_readout(self, g, key):
        if self.readout_mode == "sum":
            return dgl.sum_nodes(g, key)
        elif self.readout_mode == "max":
            return dgl.max_nodes(g, key)
        elif self.readout_mode == "mean":
            return dgl.mean_nodes(g, key)
        elif self.readout_mode == "mean+sum":
            return torch.cat([dgl.mean_nodes(g, key), dgl.sum_nodes(g, key)], dim=-1)
        else:
            # 默认 mean
            return dgl.mean_nodes(g, key)

    def _edge_readout(self, g, key):
        if self.readout_mode == "sum":
            return dgl.sum_edges(g, key)
        elif self.readout_mode == "max":
            return dgl.max_edges(g, key)
        elif self.readout_mode == "mean":
            return dgl.mean_edges(g, key)
        elif self.readout_mode == "mean+sum":
            return torch.cat([dgl.mean_edges(g, key), dgl.sum_edges(g, key)], dim=-1)
        else:
            return dgl.mean_edges(g, key)

    # ---- 前向传播 ----
    def forward(self, dataset_idx, g, h, e, training_flag, h_pos_enc=None):
        # 节点嵌入
        h = self.embedding_h(h)

        # 位置编码（如果有）
        h = self.in_feat_dropout(h)
        if self.pos_enc and (h_pos_enc is not None):
            h_pos_enc = self.embedding_pos_enc(h_pos_enc.float())
            h = h + h_pos_enc

        # 边嵌入（现在 e 的长度已与 g 的边数一致）
        e = self.embedding_e(e)

        # 骨干
        if self.backbone_type == 'gatedgcn':
            h, e = self.backbone(g, h, e)   # 同时更新边表征
        else:
            h = self.backbone(g, h)         # 仅节点表征
        g.ndata['h'] = h
        g.edata['e'] = e

        # Readout（节点+边）并拼接
        hg_node = self._graph_readout(g, 'h')
        hg_edge = self._edge_readout(g, 'e')
        hg = torch.cat([hg_node, hg_edge], dim=-1)  # (B, out_dim*2) —— 你原来的接口约定

        # MoE（可选）
        if self.moe is not None:
            hg = self.moe(hg)

        # 任务头
        attention_score = None  # 非 Transformer/注意力可视化路径时设为 None
        if self.task_type == 'cls':
            if len(self.num_tasks) == 1:
                output = self.classifier_single(hg)
                if self.num_tasks[0] > 1 and not training_flag:
                    output = torch.sigmoid(output)
                return output, attention_score
            else:
                if dataset_idx == 0:   # BACE
                    out = self.classifier1(hg)
                elif dataset_idx == 1: # BBBP
                    out = self.classifier2(hg)
                elif dataset_idx == 2: # ClinTox
                    out = self.classifier3(hg)
                    if not training_flag: out = torch.sigmoid(out)
                elif dataset_idx == 3: # SIDER
                    out = self.classifier4(hg)
                    if not training_flag: out = torch.sigmoid(out)
                else:
                    raise ValueError(f"Unknown dataset_idx {dataset_idx}")
                return out, attention_score
        else:
            if dataset_idx == 0:   # FreeSolv
                return self.regressor1(hg), attention_score
            elif dataset_idx == 1: # Lipophilicity
                return self.regressor2(hg), attention_score
            elif dataset_idx == 2: # ESOL
                return self.regressor3(hg), attention_score
            else:
                raise ValueError(f"Unknown dataset_idx {dataset_idx}")

    # ---- 损失函数（保持与你原函数一致） ----
    def loss(self, scores, targets, num_task):
        task_type = self.task_type
        if task_type == 'cls':
            if num_task == 1:
                if targets.dtype != torch.long: targets = targets.long()
                loss = nn.CrossEntropyLoss()(scores, targets)
            else:
                if targets.dtype != torch.float: targets = targets.float()
                loss = nn.BCEWithLogitsLoss()(scores, targets)
        else:
            # 回归：RMSE（注意最终你在 train 里又 sqrt 了一次，这里只给 MSE，外部再 sqrt）
            loss = nn.MSELoss()(scores, targets)
            loss = torch.sqrt(loss)
        return loss
