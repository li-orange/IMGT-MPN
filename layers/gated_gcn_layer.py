import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn

"""
    ResGatedGCN: Residual Gated Graph ConvNets
    An Experimental Study of Neural Networks for Variable Graphs (Xavier Bresson and Thomas Laurent, ICLR 2018)
    https://arxiv.org/pdf/1711.07553v2.pdf
"""

class GatedGCNLayer(nn.Module):
    """
        Param: []
    """
    def __init__(self, input_dim, output_dim, dropout, batch_norm, residual=False):
        super().__init__()
        self.in_channels = input_dim
        self.out_channels = output_dim
        self.dropout = dropout
        self.batch_norm = batch_norm
        self.residual = residual
        
        if input_dim != output_dim:
            self.residual = False
        
        self.A = nn.Linear(input_dim, output_dim, bias=True)
        self.B = nn.Linear(input_dim, output_dim, bias=True)
        self.C = nn.Linear(input_dim, output_dim, bias=True)
        self.D = nn.Linear(input_dim, output_dim, bias=True)
        self.E = nn.Linear(input_dim, output_dim, bias=True)
        self.bn_node_h = nn.BatchNorm1d(output_dim)
        self.bn_node_e = nn.BatchNorm1d(output_dim)

    def message_func(self, edges):
        Bh_j = edges.src['Bh']    
        e_ij = edges.data['Ce'] +  edges.src['Dh'] + edges.dst['Eh'] # e_ij = Ce_ij + Dhi + Ehj
        edges.data['e'] = e_ij
        return {'Bh_j' : Bh_j, 'e_ij' : e_ij}

    def reduce_func(self, nodes):
        Ah_i = nodes.data['Ah']
        Bh_j = nodes.mailbox['Bh_j']
        e = nodes.mailbox['e_ij'] 
        sigma_ij = torch.sigmoid(e) # sigma_ij = sigmoid(e_ij)
        #h = Ah_i + torch.mean( sigma_ij * Bh_j, dim=1 ) # hi = Ahi + mean_j alpha_ij * Bhj 
        h = Ah_i + torch.sum( sigma_ij * Bh_j, dim=1 ) / ( torch.sum( sigma_ij, dim=1 ) + 1e-6 )  # hi = Ahi + sum_j eta_ij/sum_j' eta_ij' * Bhj <= dense attention       
        return {'h' : h}
    
    def forward(self, g, h, e):
        
        h_in = h # 用于残差连接的输入特征
        e_in = e # 用于残差连接的边特征
        
        g.ndata['h']  = h # 将节点特征赋值给图的节点数据
        g.ndata['Ah'] = self.A(h) # 对节点特征应用线性变换A
        g.ndata['Bh'] = self.B(h) # 对节点特征应用线性变换B
        g.ndata['Dh'] = self.D(h) # 对节点特征应用线性变换D
        g.ndata['Eh'] = self.E(h) # 对节点特征应用线性变换E
        g.edata['e']  = e # 将边特征赋值给图的边数据
        g.edata['Ce'] = self.C(e) # 对边特征应用线性变换C

        g.apply_edges(fn.u_add_v('Dh', 'Eh', 'DEh')) # 计算边上的Dh + Eh，并赋值给DEh
        g.edata['e'] = g.edata['DEh'] + g.edata['Ce'] # 计算边上的新特征 e = DEh + Ce
        g.edata['sigma'] = torch.sigmoid(g.edata['e']) # 计算边上的σ = sigmoid(e)，即边的权重，用于消息传递的比例
        g.update_all(fn.u_mul_e('Bh', 'sigma', 'm'), fn.sum('m', 'sum_sigma_h')) # 计算消息传递 m = Bh * σ，并在目标节点上求和，得到sum_sigma_h
        g.update_all(fn.copy_e('sigma', 'm'), fn.sum('m', 'sum_sigma')) # 将σ复制为消息 m，并在目标节点上求和，得到sum_sigma
        g.ndata['h'] = g.ndata['Ah'] + g.ndata['sum_sigma_h'] / (g.ndata['sum_sigma'] + 1e-6) # 更新节点特征 h = Ah + sum_sigma_h / (sum_sigma + 1e-6)
        # g.update_all(self.message_func,self.reduce_func) # 注释掉的消息传递和归约函数
        h = g.ndata['h'] # 图卷积后的节点特征结果
        e = g.edata['e'] # 图卷积后的边特征结果
        
        if self.batch_norm:
            h = self.bn_node_h(h) # batch normalization  
            e = self.bn_node_e(e) # batch normalization  
        
        h = F.relu(h) # non-linear activation
        e = F.relu(e) # non-linear activation
        
        if self.residual:
            h = h_in + h # residual connection
            e = e_in + e # residual connection
        
        h = F.dropout(h, self.dropout, training=self.training)
        e = F.dropout(e, self.dropout, training=self.training)
        
        return h, e
    
    def __repr__(self):
        return '{}(in_channels={}, out_channels={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels)

    
##############################################################
#
# Additional layers for edge feature/representation analysis
#
##############################################################


class GatedGCNLayerEdgeFeatOnly(nn.Module):
    """
        Param: []
    """
    def __init__(self, input_dim, output_dim, dropout, batch_norm, residual=False):
        super().__init__()
        self.in_channels = input_dim
        self.out_channels = output_dim
        self.dropout = dropout
        self.batch_norm = batch_norm
        self.residual = residual
        
        if input_dim != output_dim:
            self.residual = False
        
        self.A = nn.Linear(input_dim, output_dim, bias=True)
        self.B = nn.Linear(input_dim, output_dim, bias=True)
        self.D = nn.Linear(input_dim, output_dim, bias=True)
        self.E = nn.Linear(input_dim, output_dim, bias=True)
        self.bn_node_h = nn.BatchNorm1d(output_dim)

    def message_func(self, edges):
        Bh_j = edges.src['Bh']    
        e_ij = edges.src['Dh'] + edges.dst['Eh'] # e_ij = Dhi + Ehj
        edges.data['e'] = e_ij
        return {'Bh_j' : Bh_j, 'e_ij' : e_ij}

    def reduce_func(self, nodes):
        Ah_i = nodes.data['Ah']
        Bh_j = nodes.mailbox['Bh_j']
        e = nodes.mailbox['e_ij'] 
        sigma_ij = torch.sigmoid(e) # sigma_ij = sigmoid(e_ij)
        h = Ah_i + torch.sum( sigma_ij * Bh_j, dim=1 ) / ( torch.sum( sigma_ij, dim=1 ) + 1e-6 )  # hi = Ahi + sum_j eta_ij/sum_j' eta_ij' * Bhj <= dense attention
        return {'h' : h}
    
    def forward(self, g, h, e):
        
        h_in = h # for residual connection
        
        g.ndata['h']  = h 
        g.ndata['Ah'] = self.A(h) 
        g.ndata['Bh'] = self.B(h) 
        g.ndata['Dh'] = self.D(h)
        g.ndata['Eh'] = self.E(h) 
        #g.update_all(self.message_func,self.reduce_func) 
        g.apply_edges(fn.u_add_v('Dh', 'Eh', 'e'))
        g.edata['sigma'] = torch.sigmoid(g.edata['e'])
        g.update_all(fn.u_mul_e('Bh', 'sigma', 'm'), fn.sum('m', 'sum_sigma_h'))
        g.update_all(fn.copy_e('sigma', 'm'), fn.sum('m', 'sum_sigma'))
        g.ndata['h'] = g.ndata['Ah'] + g.ndata['sum_sigma_h'] / (g.ndata['sum_sigma'] + 1e-6)
        h = g.ndata['h'] # result of graph convolution
        
        if self.batch_norm:
            h = self.bn_node_h(h) # batch normalization    
        
        h = F.relu(h) # non-linear activation
        
        if self.residual:
            h = h_in + h # residual connection
        
        h = F.dropout(h, self.dropout, training=self.training)
        
        return h, e
    
    def __repr__(self):
        return '{}(in_channels={}, out_channels={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels)


##############################################################


class GatedGCNLayerIsotropic(nn.Module):
    """
        Param: []
    """
    def __init__(self, input_dim, output_dim, dropout, batch_norm, residual=False):
        super().__init__()
        self.in_channels = input_dim
        self.out_channels = output_dim
        self.dropout = dropout
        self.batch_norm = batch_norm
        self.residual = residual
        
        if input_dim != output_dim:
            self.residual = False
        
        self.A = nn.Linear(input_dim, output_dim, bias=True)
        self.B = nn.Linear(input_dim, output_dim, bias=True)
        self.bn_node_h = nn.BatchNorm1d(output_dim)

    def message_func(self, edges):
        Bh_j = edges.src['Bh']
        return {'Bh_j' : Bh_j}

    def reduce_func(self, nodes):
        Ah_i = nodes.data['Ah']
        Bh_j = nodes.mailbox['Bh_j']
        h = Ah_i + torch.sum( Bh_j, dim=1 )  # hi = Ahi + sum_j Bhj
        return {'h' : h}
    
    def forward(self, g, h, e):
        
        h_in = h # for residual connection
        
        g.ndata['h']  = h 
        g.ndata['Ah'] = self.A(h) 
        g.ndata['Bh'] = self.B(h)
        #g.update_all(self.message_func,self.reduce_func) 
        g.update_all(fn.copy_u('Bh', 'm'), fn.sum('m', 'sum_h'))
        g.ndata['h'] = g.ndata['Ah'] + g.ndata['sum_h']
        h = g.ndata['h'] # result of graph convolution
        
        if self.batch_norm:
            h = self.bn_node_h(h) # batch normalization    
        
        h = F.relu(h) # non-linear activation
        
        if self.residual:
            h = h_in + h # residual connection
        
        h = F.dropout(h, self.dropout, training=self.training)
        
        return h, e
    
    def __repr__(self):
        return '{}(in_channels={}, out_channels={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels)
    

