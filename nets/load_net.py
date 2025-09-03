"""
    Utility file to select GraphNN model as
    selected by the user
"""
from nets.mpnn_net import MPNNNet
from nets.pna_net import PNANet
from nets.gated_gcn_net import GatedGCNNet
from nets.gcn_net import GCNNet
from nets.gat_net import GATNet
from nets.graphsage_net import GraphSageNet
from nets.gin_net import GINNet
from nets.mo_net import MoNet as MoNet_
from nets.mlp_net import MLPNet
from nets.gcn_att_net import GCNATTNet
from nets.gnn_moe_net import GNNWithOptionalMoE

def GNNMOE(net_params):
    return GNNWithOptionalMoE(net_params)

def GCNATT(net_params):
    return GCNATTNet(net_params)

def MPNN(net_params):
    return MPNNNet(net_params)

def PNA(net_params):
    return PNANet(net_params)

def GatedGCN(net_params):
    return GatedGCNNet(net_params)

def GCN(net_params):
    return GCNNet(net_params)

def GAT(net_params):
    return GATNet(net_params)

def GraphSage(net_params):
    return GraphSageNet(net_params)

def GIN(net_params):
    return GINNet(net_params)

def MoNet(net_params):
    return MoNet_(net_params)

def MLP(net_params):
    return MLPNet(net_params)


def gnn_model(MODEL_NAME, net_params):
    models = {
        'MPNN': MPNN,
        'PNA': PNA,
        'GCNATT': GCNATT,
        'GNNMOE': GNNMOE,
        'GatedGCN': GatedGCN,
        'GCN': GCN,
        'GAT': GAT,
        'GraphSage': GraphSage,
        'GIN': GIN,
        'MoNet': MoNet,
        'MLP': MLP
    }
        
    return models[MODEL_NAME](net_params)