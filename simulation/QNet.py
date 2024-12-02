import numpy as np
import time

import torch
import torch.nn as nn
from torch.nn import Sequential, Linear, ReLU, Transformer, Dropout
import torch.nn.functional as F
import torch.optim as optim

import torch_geometric
from torch_geometric.nn import GCNConv, SAGEConv, GATConv
from torch_geometric.nn import EdgeConv, DynamicEdgeConv, EdgePooling
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
#from torchrl.modules import NoisyLinear
from rl_utils import NoisyLinear

from CONST import (
    DUELINGMODE,NOISYMODE, KNN_AGENTS, #NUM_NODES_OF_LOCAL_GRAPH_FROM, NUM_NODES_OF_LOCAL_GRAPH_TO,
    NUM_NODES_OF_LOCAL_GRAPH_START, NUM_NODES_OF_LOCAL_GRAPH_DEST,
    DEVICE, LOCALGCNMODE, PARTIAL_AGENT_MODE,
    )

from graphDB import NODE_DATA_DIM, NODE_DATA_EACH_DIM, NODE_DATA_COMMON_DIM, POSMODE
if POSMODE:
    from graphDB import NODE_INFO_COMMON_POSX, NODE_INFO_COMMON_POSY

#from GCN import EdgeConv, DynamicEdgeConv
from CONST import (
    COMMONDIRSLA, DIRNAME, DIR_DICT, MODE_ID, TMPDIRNAME
)

DIM2=False
"""
NoisyNetを使いたくない場合はコメントアウトされている方のQNetを使う。
ただし可読性は最悪。
"""
#"""
class D_QNet(nn.Module):
    def __init__(self, state_dim:int ,action_dims:int) -> None:
        super().__init__()
        self.state_dims = state_dim
        if POSMODE:
            self.state_dims = state_dim + (action_dims+1)*KNN_AGENTS
        self.action_dims = action_dims
        
        self.dims_l1 = 1024
        self.dims_before_last = self.dims_l1
        
        self.bn1 = nn.BatchNorm1d(self.dims_l1)
        self.ln1 = nn.LayerNorm(self.dims_l1)
        
        self.l1 = nn.Linear(self.state_dims, self.dims_l1)
        nn.init.kaiming_uniform_(self.l1.weight, mode="fan_in", nonlinearity="relu")
        
        if DIM2:
            self.dims_l2 = 1024
            self.l2 = nn.Linear(self.dims_l1, self.dims_l2)
            
            self.bn2 = nn.BatchNorm1d(self.dims_l2)
            self.ln2 = nn.LayerNorm(self.dims_l2)
            
            self.dims_before_last = self.dims_l2
            nn.init.kaiming_uniform_(self.l2.weight, mode="fan_in", nonlinearity="relu")
        
        self.l_last = FactorizedNoisy(self.dims_before_last, action_dims)
        self.ladv_last = FactorizedNoisy(self.dims_before_last, action_dims)
        self.lv_last = FactorizedNoisy(self.dims_before_last, 1)

        
        #self.l_last = NoisyLinear(self.dims_before_last, action_dims)
        #self.ladv_last = NoisyLinear(self.dims_before_last, action_dims)
        #self.lv_last = NoisyLinear(self.dims_before_last, 1)
    
    #@profile
    def forward(self, x, local_edge, rows_common, rows_local):
        # print(x[0])
        #time.sleep(1)
        if POSMODE:
            x = self.preprocess(x)
        x = F.relu(self.ln1(self.l1(x)))
        if DIM2:
            x = F.relu(self.ln2(self.l2(x)))

        if DUELINGMODE:
            adv = self.ladv_last(x)
            v = self.lv_last(x)
            averagea = adv.mean(1, keepdim=True)
            return v.expand(-1, self.action_dims) + (adv - averagea.expand(-1, self.action_dims))
        else:
            x = self.l_last(x)
            return x
    
    def reset_env(self, actionlist, connect_dict, connect_to_dict):
        return
    
    def preprocess_pos(self, pos_nodes_info):
        # pos_nodes_infoがGPUにある場合、そのまま使用
        posclone = pos_nodes_info.clone().detach()

        # 最初のノードの位置を基準にして、位置情報を正規化
        posclone[:, :, 1:, :] -= posclone[:, :, 0:1, :].expand_as(posclone[:, :, 1:, :])

        # 結果テンソルの初期化
        result = torch.empty(posclone.size(0), posclone.size(1), posclone.size(2), 3, device=posclone.device)

        # ベクトルの長さと単位ベクトルを計算
        lengths = torch.norm(posclone[:, :, 1:, :], dim=3, keepdim=True)
        unit_vectors = torch.where(lengths > 0, posclone[:, :, 1:, :] / lengths, torch.zeros_like(posclone[:, :, 1:, :]))

        # 結果テンソルへの単位ベクトルと逆長さの割り当て
        result[:, :, 1:, :2] = unit_vectors
        #result[:, :, 1:, 2] = 1 / (lengths.squeeze(-1) + 1)
        result[:, :, 1:, 2] = torch.min(torch.ones_like(lengths.squeeze(-1)), lengths.squeeze(-1) / 10)


        # 最初のノードの結果をゼロに設定
        result[:, :, 0, :] = 0
        return result
    
    def preprocess(self, x):
        #print(x.shape)
        x_copy = x.clone().detach()
        x_reshape = torch.reshape(x_copy, (x_copy.shape[0], KNN_AGENTS, -1))
        x_split = torch.split(x_reshape, (NODE_DATA_DIM*(self.action_dims+1),2), dim=2)
        x_node_infos = torch.reshape(x_split[0], (x_copy.shape[0], KNN_AGENTS, (self.action_dims+1), -1))
        x_elses = x_split[1]
        x_all_node_infos = x_node_infos[:,:,:,0:NODE_DATA_DIM-2]
        x_pos = x_node_infos[:,:,:,NODE_DATA_EACH_DIM+NODE_INFO_COMMON_POSX:NODE_DATA_DIM]
        x_pos = self.preprocess_pos(x_pos)
        x_node_infos = torch.cat((x_all_node_infos, x_pos), dim=3)
        x_node_infos = torch.reshape(x_node_infos, (x_copy.shape[0], KNN_AGENTS,-1))
        x_copy = torch.cat((x_node_infos, x_elses), dim=2)

        x_copy = torch.reshape(x_copy, (x_copy.shape[0],-1))
        #print(x_copy.shape)

        return x_copy

        
class F_QNet(nn.Module):
    def __init__(self, state_dim:int ,action_dims:int) -> None:
        super().__init__()
        path = COMMONDIRSLA + DIRNAME + DIR_DICT[MODE_ID] + "/"
        path = path + TMPDIRNAME + "/" +"target_model.pt"
        #print(path)

        self.state_dims = state_dim
        if POSMODE:
            self.state_dims = state_dim + (action_dims+1)*KNN_AGENTS
        self.action_dims = action_dims

        #self.d_qnet = torch.load(path)
        self.d_qnet = D_QNet(state_dim, action_dims)
        self.d_qnet.load_state_dict(torch.load(path))

        for param in self.d_qnet.parameters():
            param.requires_grad = False
    
    def forward(self, x):
        x = self.d_qnet(x)
        return x

    def reset_env(self, actionlist, connect_dict, connect_to_dict):
        return

#"""

class FactorizedNoisy(nn.Module):
    def __init__(self, in_features, out_features):
        super(FactorizedNoisy, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        # 学習パラメータを生成
        self.u_w = nn.Parameter(torch.Tensor(out_features, in_features))
        self.sigma_w  = nn.Parameter(torch.Tensor(out_features, in_features))
        self.u_b = nn.Parameter(torch.Tensor(out_features))
        self.sigma_b = nn.Parameter(torch.Tensor(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        # 初期値設定
        stdv = 1. / np.sqrt(self.u_w.size(1))
        self.u_w.data.uniform_(-stdv, stdv)
        self.u_b.data.uniform_(-stdv, stdv)

        initial_sigma = 0.5 * stdv
        self.sigma_w.data.fill_(initial_sigma)
        self.sigma_b.data.fill_(initial_sigma)

    def forward(self, x):
        # 毎回乱数を生成
        rand_in = self._f(torch.randn(1, self.in_features, device=self.u_w.device))
        rand_out = self._f(torch.randn(self.out_features, 1, device=self.u_w.device))
        epsilon_w = torch.matmul(rand_out, rand_in)
        epsilon_b = rand_out.squeeze()

        w = self.u_w + self.sigma_w * epsilon_w
        b = self.u_b + self.sigma_b * epsilon_b
        return F.linear(x, w, b)

    def _f(self, x):
        return torch.sign(x) * torch.sqrt(torch.abs(x))

SAME_GNN_MODEL = True

#"""
class G_QNet(nn.Module):
    def __init__(self, state_dim:int ,action_dims:int) -> None:
        super().__init__()
        self.state_dims = state_dim
        self.action_dims = action_dims

        self.node_data_dim = NODE_DATA_DIM
        self.gcn_node_data_before_dim = NODE_DATA_COMMON_DIM
        self.gcn_node_data_middle_dim = 3
        self.gcn_node_data_after_dim = 2
        if POSMODE:
            self.node_data_dim = NODE_DATA_DIM + 1
            self.gcn_node_data_before_dim = NODE_DATA_COMMON_DIM + 1
            self.gcn_node_data_after_dim = NODE_DATA_COMMON_DIM + 1
            #print(self.node_data_dim)
            
        self.startnode_dims = NODE_DATA_DIM * NUM_NODES_OF_LOCAL_GRAPH_START
        self.destnode_dims = NODE_DATA_DIM * NUM_NODES_OF_LOCAL_GRAPH_DEST
        self.fromnodes = NUM_NODES_OF_LOCAL_GRAPH_START
        self.tonodes = NUM_NODES_OF_LOCAL_GRAPH_DEST
        self.elsedim = 2
        
        self.node_data_dim_after_gnn = NODE_DATA_EACH_DIM + self.gcn_node_data_after_dim

        self.singledims = self.elsedim + self.node_data_dim_after_gnn * (self.action_dims + 1)
        self.connect_dims = self.singledims * KNN_AGENTS
        
        self.gcn_conv = GCNConv(self.gcn_node_data_before_dim, self.gcn_node_data_middle_dim)
        self.gcn_conv2 = GCNConv(self.gcn_node_data_middle_dim, self.gcn_node_data_middle_dim)
        self.gcn_conv3 = GCNConv(self.gcn_node_data_middle_dim, self.gcn_node_data_after_dim)
        #self.bn_gcn = nn.BatchNorm1d(self.gcn_node_data_after_dim)
        
        
        self.sage_conv = SAGEConv(self.gcn_node_data_before_dim, self.gcn_node_data_middle_dim)
        self.sage_conv2 = SAGEConv(self.gcn_node_data_middle_dim, self.gcn_node_data_after_dim)
        
        self.gat_conv = GATConv(self.gcn_node_data_before_dim, self.gcn_node_data_middle_dim)
        self.gat_conv2 = GATConv(self.gcn_node_data_middle_dim, self.gcn_node_data_after_dim)
        
        self.ln_gcn = torch_geometric.nn.LayerNorm(self.gcn_node_data_middle_dim)
        self.ln_gcn2 = torch_geometric.nn.LayerNorm(self.gcn_node_data_middle_dim)
        self.ln_gcn3 = torch_geometric.nn.LayerNorm(self.gcn_node_data_after_dim)
        
        for_edge_conv = Sequential(\
            Linear(2 * self.gcn_node_data_before_dim, self.gcn_node_data_before_dim), \
                ReLU(), \
                    Linear(self.gcn_node_data_before_dim, self.gcn_node_data_after_dim))
        
        for_edge_conv2 = Sequential(\
            Linear(2 * self.gcn_node_data_after_dim, self.gcn_node_data_after_dim), \
                ReLU(), \
                    Linear(self.gcn_node_data_after_dim, self.gcn_node_data_after_dim))
        self.edge_conv = EdgeConv(for_edge_conv)
        self.edge_conv2 = EdgeConv(for_edge_conv2)
        #self.dynamic_edge_conv = DynamicEdgeConv(for_edge_conv, k=(self.action_dims))
        #self.dynamic_edge_conv2 = DynamicEdgeConv(for_edge_conv2, k=(self.action_dims))
        
        self.dropout = Dropout(0.5)
        #self.transformer = Transformer(nhead=16, num_encoder_layers=12, batch_first=True)
        #self.gat = GATConv(self.gcn_node_data_before_dim, self.gcn_node_data_after_dim)
        ####
        
        
        self.bn_connect = nn.BatchNorm1d(self.connect_dims)
        self.ln_connect = nn.LayerNorm(self.connect_dims)
        
        
        self.dims_l1 = 1024
        
        self.bn1 = nn.BatchNorm1d(self.dims_l1)
        self.ln1 = nn.LayerNorm(self.dims_l1)
        
        self.dims_before_last = self.dims_l1
        self.l1 = nn.Linear(self.connect_dims, self.dims_l1)
        nn.init.kaiming_uniform_(self.l1.weight, mode="fan_in", nonlinearity="relu")

        self.dims_l2 = 1024
        
        self.bn2 = nn.BatchNorm1d(self.dims_l2)
        self.ln2 = nn.LayerNorm(self.dims_l2)
        
        self.l2 = nn.Linear(self.dims_l1, self.dims_l2)
        nn.init.kaiming_uniform_(self.l2.weight, mode="fan_in", nonlinearity="relu")


        self.dims_before_last = self.dims_l1
        self.l_last = FactorizedNoisy(self.dims_before_last, action_dims)
        self.ladv_last = FactorizedNoisy(self.dims_before_last, action_dims)
        self.lv_last = FactorizedNoisy(self.dims_before_last, 1)
        
        self.flatten = nn.Flatten()
    
    #@profile #GNN 85% of savetraining.py.This is most important part.
    #def forward(self, x, connect_dict_batch):
    
    #@profile
    def forward(self, x, local_edge, rows_common, each_info):
        #### Graph #####
        #print(x.shape)
        batch_num = x.shape[0]
        x_reshape = torch.reshape(x, (batch_num, KNN_AGENTS, -1))
        
        #x_split = torch.split(x_reshape, (self.startnode_dims, self.destnode_dims, self.fromnodes, self.tonodes, self.elsedim), dim=2)
        #x_start_infos = torch.reshape(x_split[0], (batch_num, KNN_AGENTS, self.fromnodes, -1))
        #x_dest_infos = torch.reshape(x_split[1], (batch_num, KNN_AGENTS, self.tonodes, -1))
        #x_elses = x_split[4]
        
        x_split = torch.split(x_reshape, (self.startnode_dims, self.destnode_dims, self.elsedim), dim=2)
        x_start_infos = torch.reshape(x_split[0], (batch_num, KNN_AGENTS, self.fromnodes, -1))
        x_dest_infos = torch.reshape(x_split[1], (batch_num, KNN_AGENTS, self.tonodes, -1))
        x_elses = x_split[2]
        
        x_all_start_infos = x_start_infos[:,:,:,NODE_DATA_EACH_DIM:NODE_DATA_DIM]
        x_all_dest_infos = x_dest_infos[:,:,:,NODE_DATA_EACH_DIM:NODE_DATA_DIM]
        #x_each_start_infos = x_start_infos[:,:,:,0:NODE_DATA_EACH_DIM]
        #x_each_dest_infos = x_dest_infos[:,:,:,0:NODE_DATA_EACH_DIM]
        
        if POSMODE:
            #print(x_start_infos.shape)
            x_all_start_infos = x_start_infos[:,:,:,NODE_DATA_EACH_DIM:NODE_DATA_DIM-2]
            x_all_dest_infos = x_dest_infos[:,:,:,NODE_DATA_EACH_DIM:NODE_DATA_DIM-2]

            x_all_start_infos_poss = x_start_infos[:,:,:,NODE_DATA_EACH_DIM+NODE_INFO_COMMON_POSX:NODE_DATA_DIM]
            x_all_dest_infos_poss = x_dest_infos[:,:,:,NODE_DATA_EACH_DIM+NODE_INFO_COMMON_POSX:NODE_DATA_DIM]
            
            x_all_infos_pos = torch.cat((x_all_start_infos_poss, x_all_dest_infos_poss), dim=2)
            x_all_infos_pos = self.preprocess_pos(x_all_infos_pos)
            x_all_start_infos_pos_process, x_all_dest_infos_pos_process = torch.split(x_all_infos_pos, [self.fromnodes, self.tonodes], dim=2)

            x_all_start_infos = torch.cat((x_all_start_infos, x_all_start_infos_pos_process), dim=3)
            x_all_dest_infos = torch.cat((x_all_dest_infos, x_all_dest_infos_pos_process), dim=3)
            
        x_infos_batch = torch.reshape(torch.cat([x_all_start_infos, x_all_dest_infos], dim=2), (batch_num, KNN_AGENTS*(NUM_NODES_OF_LOCAL_GRAPH_START+NUM_NODES_OF_LOCAL_GRAPH_DEST), -1))
        #x_each_infos = torch.cat([x_each_start_infos, x_each_dest_infos], dim=2)  # [KNN_AGENTS, M] where M is the combined size
        # GNNに通すためのdataset 作成前処理
        x_infos = []
        #"""
        datasets = [self.make_gnn_dataset(x_infos_batch[b], local_edge[b]) for b in range(batch_num)]
        dataset_loader = DataLoader(datasets, batch_size=batch_num, shuffle=False)
        
        #print(x_infos_batch.shape)
        for data in dataset_loader: # GNN処理 # GNN 8.5%
            x_data = data.x
            #print(x_data.shape)
            
            #x_data = self.state_graph_l1(x_data, data.edge_index)
            #x_data = F.relu(x_data)
            ###
            #x_data = self.edge_conv(x_data, data.edge_index)
            x_data = self.gcn_conv(x_data, data.edge_index)
            #x_data = self.sage_conv(x_data, data.edge_index)
            #x_data = self.gat_conv(x_data, data.edge_index)
            #x_data = self.edge_conv(x_data)
            x_data = self.ln_gcn(x_data)
            x_data = F.relu(x_data)
            x_data = self.dropout(x_data)

            #x_data = self.edge_conv2(x_data, data.edge_index)
            #x_data = self.gcn_conv2(x_data, data.edge_index)
            #x_data = self.sage_conv2(x_data, data.edge_index)
            #x_data = self.gat_conv2(x_data, data.edge_index)
            #x_data = self.edge_conv(x_data)
            #x_data = self.ln_gcn2(x_data)
            #x_data = F.relu(x_data)
            #x_data = self.dropout(x_data)
            
            
            #x_data = self.edge_conv2(x_data, data.edge_index)
            x_data = self.gcn_conv3(x_data, data.edge_index)
            #x_data = self.sage_conv2(x_data, data.edge_index)
            #x_data = self.gat_conv2(x_data, data.edge_index)
            #x_data = self.edge_conv(x_data)
            x_data = self.ln_gcn3(x_data)
            x_data = F.relu(x_data)
            #x_data = self.dropout(x_data)
            
            #x_data = self.edge_conv2(x_data)
            #x_data = F.relu(x_data)
            ###
            x_infos.append(x_data)
        #"""
        

        #"""
        x_all_infos_after_gnn = torch.cat(x_infos)
        x_all_infos = x_all_infos_after_gnn.reshape((batch_num, KNN_AGENTS*(self.fromnodes+self.tonodes), -1))
        #"""
        
        rows_common = torch.tensor(np.concatenate(rows_common).reshape(batch_num, KNN_AGENTS, -1)).to(DEVICE)
        #rows_local = torch.tensor(np.concatenate(rows_local).reshape(batch_num, KNN_AGENTS, -1)).to(DEVICE)
        
        x_all_info_flats = torch.flatten(
            torch.stack(
                [
                    x_all_infos[batch][rows_common[batch]] for batch in range(batch_num)
                ]
            ), start_dim=2
        )

        x_each_info_flats = torch.flatten(each_info, start_dim=2)
        x = torch.flatten(torch.cat([x_all_info_flats, x_elses, x_each_info_flats], dim=2), start_dim=1)
        
        #### Graph End #####
        #x = F.dropout(x, p=0.5)

        #x = F.relu(self.l1(x))
        x = F.relu(self.ln1(self.l1(x)))
        #x = F.relu(self.ln2(self.l2(x)))
        #x = F.relu(self.l2(x))
        if DUELINGMODE:
            adv = self.ladv_last(x)
            v = self.lv_last(x)
            averagea = adv.mean(1, keepdim=True)
            return v.expand(-1, self.action_dims) + (adv - averagea.expand(-1, self.action_dims))
        else:
            x = self.l_last(x)
            return x
    
    #@profile
    def make_gnn_dataset(self, nodes_info, local_edge):
        edges_index = torch.tensor(local_edge, dtype=torch.int64).to(DEVICE)
        return Data(x=nodes_info, edge_index=edges_index)
    
    #"""
    def preprocess_pos(self, pos_nodes_info):
        # pos_nodes_infoがGPUにある場合、そのまま使用
        posclone = pos_nodes_info.clone().detach()

        # 最初のノードの位置を基準にして、位置情報を正規化
        posclone[:, :, 1:, :] -= posclone[:, :, 0:1, :].expand_as(posclone[:, :, 1:, :])

        # 結果テンソルの初期化
        result = torch.empty(posclone.size(0), posclone.size(1), posclone.size(2), 3, device=posclone.device)

        # ベクトルの長さと単位ベクトルを計算
        lengths = torch.norm(posclone[:, :, 1:, :], dim=3, keepdim=True)
        unit_vectors = torch.where(lengths > 0, posclone[:, :, 1:, :] / lengths, torch.zeros_like(posclone[:, :, 1:, :]))

        # 結果テンソルへの単位ベクトルと逆長さの割り当て
        result[:, :, 1:, :2] = unit_vectors
        #result[:, :, 1:, 2] = 1 / (lengths.squeeze(-1) + 1)
        result[:, :, 1:, 2] = torch.min(torch.ones_like(lengths.squeeze(-1)), lengths.squeeze(-1) / 10)

        # 最初のノードの結果をゼロに設定
        result[:, :, 0, :] = 0

        return result
        
    
    def reset_env(self, actionlist, connect_dict, connect_to_dict):
        #self.actionlist = actionlist
        #self.connect_dict = connect_dict
        #self.connect_to_dict = connect_to_dict
        return
    
#"""
#class QNet(F_QNet):
#    def __init__(self, state_dim:int ,action_dims:int) -> None:
#        super().__init__(state_dim ,action_dims)
#"""
if PARTIAL_AGENT_MODE:
    class QNet(D_QNet):
        def __init__(self, state_dim:int ,action_dims:int) -> None:
            super().__init__(state_dim ,action_dims)
elif LOCALGCNMODE:
    class QNet(G_QNet):
        def __init__(self, state_dim:int ,action_dims:int) -> None:
            super().__init__(state_dim, action_dims)
#"""