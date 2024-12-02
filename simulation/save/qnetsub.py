import numpy as np
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch_geometric.nn import GCNConv, SAGEConv
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from CONST import (
    DUELINGMODE,NOISYMODE, KNN_AGENTS, #NUM_NODES_OF_LOCAL_GRAPH_FROM, NUM_NODES_OF_LOCAL_GRAPH_TO,
    NUM_NODES_OF_LOCAL_GRAPH_START, NUM_NODES_OF_LOCAL_GRAPH_DEST,
    DEVICE
    )
from map_maker import (
    CONNECTDICT, ACTION_LIST, CONNECT_TO_DICT
)
from graphDB import NODE_DATA_DIM, NODE_DATA_EACH_DIM, NODE_DATA_COMMON_DIM

DIM2=False
"""
NoisyNetを使いたくない場合はコメントアウトされている方のQNetを使う。
ただし可読性は最悪。
"""
#"""
class QNet(nn.Module):
    def __init__(self, state_dim:int ,action_dims:int) -> None:
        super().__init__()
        self.state_dims = state_dim
        self.action_dims = action_dims
        
        self.dims_l1 = 1024
        self.dims_before_last = self.dims_l1
        self.l1 = nn.Linear(self.state_dims, self.dims_l1)
        nn.init.kaiming_uniform_(self.l1.weight, mode="fan_in", nonlinearity="relu")
        
        if DIM2:
            self.dims_l2 = 1024
            self.l2 = nn.Linear(self.dims_l1, self.dims_l2)
            self.dims_before_last = self.dims_l2
            nn.init.kaiming_uniform_(self.l2.weight, mode="fan_in", nonlinearity="relu")
        
        self.l_last = FactorizedNoisy(self.dims_before_last, action_dims)
        self.ladv_last = FactorizedNoisy(self.dims_before_last, action_dims)
        self.lv_last = FactorizedNoisy(self.dims_before_last, 1)
    
    #@profile
    def forward(self, x):
        x = F.relu(self.l1(x))
        if DIM2:
            x = F.relu(self.l2(x))

        if DUELINGMODE:
            adv = self.ladv_last(x)
            v = self.lv_last(x)
            averagea = adv.mean(1, keepdim=True)
            return v.expand(-1, self.action_dims) + (adv - averagea.expand(-1, self.action_dims))
        else:
            x = self.l_last(x)
            return x
#"""
"""
class QNet(nn.Module):
    def __init__(self, state_dim:int ,action_dims:int) -> None:
        super().__init__()
        self.state_dims = state_dim
        self.action_dims = action_dims
        
        self.dims_l1 = 512
        self.dims_before_last = self.dims_l1
        self.l1 = nn.Linear(self.state_dims, self.dims_l1)
        if DIM2:
            self.dims_l2 = 512
            self.l2 = nn.Linear(self.dims_l1, self.dims_l2)
            self.dims_before_last = self.dims_l2
            nn.init.kaiming_uniform_(self.l2.weight, mode="fan_in", nonlinearity="relu")
        #'''
        if NOISYMODE:
            self.l_last = FactorizedNoisy(self.dims_before_last, action_dims)
            self.ladv_last = FactorizedNoisy(self.dims_before_last, action_dims)
            self.lv_last = FactorizedNoisy(self.dims_before_last, 1)
        else:
            self.l_last = nn.Linear(self.dims_before_last, action_dims)
            self.ladv_last = nn.Linear(self.dims_before_last, action_dims)
            self.lv_last = nn.Linear(self.dims_before_last, 1)
        #'''
        nn.init.kaiming_uniform_(self.l1.weight, mode="fan_in", nonlinearity="relu")
        
        if NOISYMODE:
            pass
        else:
            nn.init.kaiming_uniform_(self.l_last.weight, mode="fan_in", nonlinearity="relu")
            nn.init.kaiming_uniform_(self.ladv_last.weight, mode="fan_in", nonlinearity="relu")
            nn.init.kaiming_uniform_(self.lv_last.weight, mode="fan_in", nonlinearity="relu")
    
    def forward(self, x):
        x = F.relu(self.l1(x))
        if DIM2:
            x = F.relu(self.l2(x))

        if DUELINGMODE:
            adv = self.ladv_last(x)
            v = self.lv_last(x)
            averagea = adv.mean(1, keepdim=True)
            return v.expand(-1, self.action_dims) + (adv - averagea.expand(-1, self.action_dims))
        else:
            x = self.l_last(x)
            return x
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

"""
class LocalGraphEmbedding(nn.Module):
    def __init__(self, input_dim:int ,output_dim:int) -> None:
        super().__init__()
        self.graph_connect = CONNECTDICT
        self.conv1 = GCNConv(input_dim, output_dim)
    
    def forward(self, x_local_graphinfo, x_local_node_id):
        batch_num = x_local_graphinfo.shape[0]
        x_local_edge = self.make_localedge(x_local_node_id, batch_num)
        x_local_graphinfo = x_local_graphinfo.reshape((batch_num, len(x_local_node_id[0]), NODE_DATA_DIM))
        print(f"x_local_graphinfo.shape{x_local_graphinfo.shape}")
        print(f"edge{x_local_edge.shape}")
        x = self.conv1(x_local_graphinfo, x_local_edge)
        x = F.relu(x)
        return x
    
    def make_localedge(self, nodes_float, batch_num):
        #multi_edges = []
        nodes = torch.tensor(nodes_float, dtype=torch.int32).numpy()
        print(nodes)
        
        multi_adjacencymatrix = []
        for i in range(batch_num):
            #edges_from = []
            #edges_to = []
            adjacencymatrix = np.zeros((len(nodes[i]), len(nodes[i])), dtype=int)
            for start_node in nodes[i]:
                for dest_node in self.graph_connect[start_node]:
                    if dest_node in nodes[i]:
                        start_idx = np.where(nodes[i]==start_node)[0][0]
                        dest_idx = np.where(nodes[i]==dest_node)[0][0]
                        #edges_from.append(np.where(nodes[i]==start_node)[0][0])
                        #edges_to.append(np.where(nodes[i]==dest_node)[0][0])
                        adjacencymatrix[start_idx][dest_idx] += 1
            multi_adjacencymatrix.append(torch.tensor(adjacencymatrix,dtype=torch.int32))

            #edges = torch.tensor([edges_from, edges_to], dtype=torch.int32)
            #multi_edges.append(edges)
        #multi_edges = torch.tensor(multi_edges)
        #print(multi_edges)
        multi_adjacencymatrix = torch.cat(multi_adjacencymatrix, dim=1)
        multi_adjacencymatrix = multi_adjacencymatrix.reshape((batch_num,len(nodes[i]), len(nodes[i])))
        print(f"mat{multi_adjacencymatrix.shape}")
        return multi_adjacencymatrix

class ReverseLocalGraphEmbedding(nn.Module):
    def __init__(self, input_dim:int ,output_dim:int) -> None:
        super().__init__()
        self.graph_connect = CONNECT_TO_DICT
        self.conv1 = GCNConv(input_dim, output_dim)
    
    def forward(self, x_local_graphinfo, x_local_node_id):
        batch_num = x_local_graphinfo.shape[0]

        x_local_edge = self.make_localedge(x_local_node_id, batch_num)
        x = self.conv1(x_local_graphinfo, x_local_edge)
        x = F.relu(x)
        return x
    
    def make_localedge(self, nodes, batch_num):
        multi_edges = []
        print(nodes)
        multi_adjacencymatrix = []
        for i in range(batch_num):
            #edges_from = []
            #edges_to = []
            adjacencymatrix = np.zeros((len(nodes[i]), len(nodes[i])), dtype=int)
            for start_node in nodes[i]:
                for dest_node in self.graph_connect[start_node]:
                    if dest_node in nodes[i]:
                        #edges_from.append(np.where(nodes[i]==start_node)[0][0])
                        #edges_to.append(np.where(nodes[i]==dest_node)[0][0])
                        adjacencymatrix[start_node][dest_node] += 1
            multi_adjacencymatrix.append(torch.tensor(adjacencymatrix,dtype=torch.int32))

            #edges = torch.tensor([edges_from, edges_to], dtype=torch.int32)
            #multi_edges.append(edges)
        #multi_edges = torch.tensor(multi_edges)
        #print(multi_edges)
        multi_adjacencymatrix = torch.cat(multi_adjacencymatrix)
        print(f"mat{multi_adjacencymatrix.shape}")
        return multi_adjacencymatrix
"""

class OLD_LocalGraphEmbedding(nn.Module):
    def __init__(self, input_dim:int ,output_dim:int, connect:dict) -> None:
        super().__init__()
        self.graph_connect = connect
        #self.conv1 = GCNConv(input_dim, output_dim)
        self.conv1 = SAGEConv(input_dim, output_dim)
    
    #@profile
    def forward(self, x_local_graphinfo, x_local_node_id, connect):
        x_local_edge = self.make_localedge(x_local_node_id, connect)
        #print(x_local_graphinfo)
        #print(x_local_edge)
        x = self.conv1(x_local_graphinfo, x_local_edge)
        x = F.relu(x)
        return x
    
    #"""
    #@profile
    def make_localedge(self, nodes_float, connect):
        nodes = nodes_float.clone().detach().to(torch.int32).to('cpu').numpy()
        edges_from = []
        edges_to = []
        #print(f"model:{len(self.graph_connect)}")
        for start_node in nodes:
            for dest_node in connect[start_node]:
                if dest_node in nodes:
                    start_idx = np.where(nodes==start_node)[0][0]
                    dest_idx = np.where(nodes==dest_node)[0][0]
                    edges_from.append(start_idx)
                    edges_to.append(dest_idx)
        edges = torch.tensor([edges_from, edges_to], dtype=torch.int64).to(DEVICE) # for Graph SAGE int64 (not int32)
        return edges
    #"""
    
    """
    #@profile
    def make_localedge(self, nodes_float):
        nodes = nodes_float.clone().detach().to(torch.int32)
        edges_from = []
        edges_to = []
        for start_node in nodes:
            for dest_node in self.graph_connect[start_node.item()]:
                if dest_node in nodes:
                    start_idx = torch.where(nodes==start_node)[0][0].item()
                    dest_idx = torch.where(nodes==dest_node)[0][0].item()
                    edges_from.append(start_idx)
                    edges_to.append(dest_idx)
        edges = torch.tensor([edges_from, edges_to], dtype=torch.int32).to(DEVICE)
        return edges
    """
SAME_GNN_MODEL = True   
"""
class QNet(nn.Module):
    def __init__(self, state_dim:int ,action_dims:int) -> None:
        super().__init__()
        self.state_dims = state_dim
        self.action_dims = action_dims

        self.node_data_dim = NODE_DATA_DIM
        self.startnode_dims = NODE_DATA_DIM * NUM_NODES_OF_LOCAL_GRAPH_START
        self.destnode_dims = NODE_DATA_DIM * NUM_NODES_OF_LOCAL_GRAPH_DEST
        self.fromnodes = NUM_NODES_OF_LOCAL_GRAPH_START
        self.tonodes = NUM_NODES_OF_LOCAL_GRAPH_DEST
        self.elsedim = 2

        self.singledims = self.elsedim + self.node_data_dim * (self.action_dims + 1)
        self.connect_dims = self.singledims * KNN_AGENTS
        #print(len(CONNECTDICT))
        #if SAME_GNN_MODEL:
        #    self.state_graph_l1 = LocalGraphEmbedding(input_dim=self.node_data_dim, output_dim=self.node_data_dim, connect=CONNECTDICT) 
            #self.dest_graph_l1 = LocalGraphEmbedding(input_dim=self.node_data_dim, output_dim=self.node_data_dim, connect=CONNECT_TO_DICT)
            #self.dest_graph_l1 = LocalGraphEmbedding(input_dim=self.node_data_dim, output_dim=self.node_data_dim, connect=CONNECTDICT)
        #    self.dest_graph_l1 = self.state_graph_l1
        #else:
        #    self.state_graph_l1 = [LocalGraphEmbedding(input_dim=self.node_data_dim, output_dim=self.node_data_dim, connect=CONNECTDICT) for _ in range(KNN_AGENTS)]
        #    self.dest_graph_l1 = [LocalGraphEmbedding(input_dim=self.node_data_dim, output_dim=self.node_data_dim, connect=CONNECT_TO_DICT) for _ in range(KNN_AGENTS)]
            #self.dest_graph_l1 = [LocalGraphEmbedding(input_dim=self.node_data_dim, output_dim=self.node_data_dim, connect=CONNECTDICT) for _ in range(KNN_AGENTS)]

        if SAME_GNN_MODEL:
            self.state_graph_l1 = LocalGraphEmbedding(input_dim=self.node_data_dim, output_dim=self.node_data_dim, connect=CONNECTDICT) 
            self.dest_graph_l1 = self.state_graph_l1
        else:
            self.state_graph_l1 = [LocalGraphEmbedding(input_dim=self.node_data_dim, output_dim=self.node_data_dim) for _ in range(KNN_AGENTS)]
            self.dest_graph_l1 = [LocalGraphEmbedding(input_dim=self.node_data_dim, output_dim=self.node_data_dim) for _ in range(KNN_AGENTS)]

        
        self.dims_l1 = 512
        self.dims_before_last = self.dims_l1
        self.l1 = nn.Linear(self.connect_dims, self.dims_l1)
        nn.init.kaiming_uniform_(self.l1.weight, mode="fan_in", nonlinearity="relu")
        
        self.l_last = FactorizedNoisy(self.dims_before_last, action_dims)
        self.ladv_last = FactorizedNoisy(self.dims_before_last, action_dims)
        self.lv_last = FactorizedNoisy(self.dims_before_last, 1)
        
        self.flatten = nn.Flatten()
    
    def find_indices(self, actionlist, nodes_float):
        indices = []
        
        nodes = nodes_float.clone().detach().to(torch.int32).to('cpu').numpy()
        for item in actionlist[nodes[0]]:
            if item in nodes:
                indices.append(np.where(nodes==item)[0][0])
        return indices

    #@profile
    def forward(self, x):
        #### Graph #####
        #print(x.shape)
        batch_num = x.shape[0]
        x_reshape = torch.reshape(x, (batch_num, KNN_AGENTS, -1))
        #print(x_reshape.shape)
        x_split = torch.split(x_reshape, (self.startnode_dims,self.destnode_dims,self.fromnodes,self.tonodes, self.elsedim), dim=2)
        x_start_infos = torch.reshape(x_split[0], (batch_num, KNN_AGENTS, self.fromnodes, -1))
        x_dest_infos = torch.reshape(x_split[1], (batch_num, KNN_AGENTS, self.tonodes, -1))
        x_start_graphs = x_split[2]
        x_dest_graphs = x_split[3]
        x_elses = x_split[4]

        #print(x_start_infos.shape)
        #print(x_dest_infos.shape)
        #print(x_start_graphs.shape)
        #print(x_dest_graphs.shape)
        #print(x_elses.shape)

        x_feat = []
        for batch in range(batch_num):
            x_start_info = x_start_infos[batch]
            x_dest_info = x_dest_infos[batch]
            x_start_graph = x_start_graphs[batch]
            x_dest_graph = x_dest_graphs[batch]
            x_else = x_elses[batch]
            x_tmp_feat = []

            #print(x_start_info.shape)
            #print(x_dest_info.shape)
            #print(x_start_graph.shape)
            #print(x_dest_graph.shape)
            for robot in range(KNN_AGENTS):
                x_start_info_robot = x_start_info[robot]
                x_dest_info_robot = x_dest_info[robot]
                x_start_graph_robot = x_start_graph[robot]
                x_dest_graph_robot = x_dest_graph[robot]
                x_else_robot = x_else[robot]
                
                #if robot == 0:
                #    print(f"qnet_start{x_start_info_robot}")
                #    print(f"qnet_dest{x_dest_info_robot}")
                #    print(f"qnet_start_g{x_start_graph_robot}")
                #    print(f"qnet_dest_g{x_dest_graph_robot}")

                #if SAME_GNN_MODEL:
                #    x_start_info_robot = self.state_graph_l1(x_start_info_robot, x_start_graph_robot)
                #    x_dest_info_robot = self.dest_graph_l1(x_dest_info_robot, x_dest_graph_robot)
                #else:
                #    x_start_info_robot = self.state_graph_l1[robot](x_start_info_robot, x_start_graph_robot)
                #    x_dest_info_robot = self.dest_graph_l1[robot](x_dest_info_robot, x_dest_graph_robot)
                
                if SAME_GNN_MODEL:
                    x_start_info_robot = self.state_graph_l1(x_start_info_robot, x_start_graph_robot, CONNECTDICT)
                    x_dest_info_robot = self.dest_graph_l1(x_dest_info_robot, x_dest_graph_robot, CONNECTDICT)
                else:
                    x_start_info_robot = self.state_graph_l1[robot](x_start_info_robot, x_start_graph_robot, CONNECTDICT)
                    x_dest_info_robot = self.dest_graph_l1[robot](x_dest_info_robot, x_dest_graph_robot, CONNECTDICT)
                
                rows = self.find_indices(ACTION_LIST, x_start_graph_robot)
                #print(torch.flatten(x_start_info_robot[rows]).shape)
                #print(x_dest_info_robot[0].shape)
                #print(x_else_robot.shape)
                x_tmp_feat.append(torch.cat([torch.flatten(x_start_info_robot[rows]), x_dest_info_robot[0], x_else_robot]))
            x_tmp_feat = torch.cat(x_tmp_feat)
            #print(x_tmp_feat.shape)
            x_feat.append(x_tmp_feat)
            
            #print("end")
        #print("for clear")
        x = torch.cat(x_feat)
        x = torch.reshape(x, (batch_num, -1))
        #print(x.shape)

        #### Graph End #####

        x = F.relu(self.l1(x))
        if DIM2:
            x = F.relu(self.l2(x))

        if DUELINGMODE:
            adv = self.ladv_last(x)
            v = self.lv_last(x)
            averagea = adv.mean(1, keepdim=True)
            return v.expand(-1, self.action_dims) + (adv - averagea.expand(-1, self.action_dims))
        else:
            x = self.l_last(x)
            return x
#"""

class LocalGraphEmbedding(nn.Module):
    def __init__(self, input_dim:int ,output_dim:int, connect:dict) -> None:
        super().__init__()
        self.graph_connect = connect
        #self.conv1 = GCNConv(input_dim, output_dim)
        self.conv1 = SAGEConv(input_dim, output_dim)
    
    #@profile
    def forward(self, x_local_graphinfo, x_local_node_id, connect):
        x_local_edge = self.make_localedge(x_local_node_id, connect)
        x = self.conv1(x_local_graphinfo, x_local_edge)
        x = F.relu(x)
        return x
    
#"""
class QNet(nn.Module):
    def __init__(self, state_dim:int ,action_dims:int) -> None:
        super().__init__()
        self.state_dims = state_dim
        self.action_dims = action_dims

        self.node_data_dim = NODE_DATA_DIM
        self.gcn_node_data_dim = NODE_DATA_COMMON_DIM
        self.startnode_dims = NODE_DATA_DIM * NUM_NODES_OF_LOCAL_GRAPH_START
        self.destnode_dims = NODE_DATA_DIM * NUM_NODES_OF_LOCAL_GRAPH_DEST
        self.fromnodes = NUM_NODES_OF_LOCAL_GRAPH_START
        self.tonodes = NUM_NODES_OF_LOCAL_GRAPH_DEST
        self.elsedim = 2

        self.singledims = self.elsedim + self.node_data_dim * (self.action_dims + 1)
        self.connect_dims = self.singledims * KNN_AGENTS
        
        if SAME_GNN_MODEL:
            #self.state_graph_l1 = LocalGraphEmbedding(input_dim=self.node_data_dim, output_dim=self.node_data_dim, connect=CONNECTDICT) 
            self.state_graph_l1 = GCNConv(self.gcn_node_data_dim, self.gcn_node_data_dim)
            self.dest_graph_l1 = self.state_graph_l1
        #else:
        #    self.state_graph_l1 = [LocalGraphEmbedding(input_dim=self.node_data_dim, output_dim=self.node_data_dim) for _ in range(KNN_AGENTS)]
        #    self.dest_graph_l1 = [LocalGraphEmbedding(input_dim=self.node_data_dim, output_dim=self.node_data_dim) for _ in range(KNN_AGENTS)]

        
        self.dims_l1 = 512
        self.dims_before_last = self.dims_l1
        self.l1 = nn.Linear(self.connect_dims, self.dims_l1)
        nn.init.kaiming_uniform_(self.l1.weight, mode="fan_in", nonlinearity="relu")
        
        self.l_last = FactorizedNoisy(self.dims_before_last, action_dims)
        self.ladv_last = FactorizedNoisy(self.dims_before_last, action_dims)
        self.lv_last = FactorizedNoisy(self.dims_before_last, 1)
        
        self.flatten = nn.Flatten()
    
    
    #def find_indices(self, actionlist, nodes_float):
    #    indices = []
        
    #    nodes = nodes_float.clone().detach().to(torch.int32).to('cpu').numpy()
    #    for item in actionlist[nodes[0]]:
    #        if item in nodes:
    #            indices.append(np.where(nodes==item)[0][-1]) # because of using enumerate in make_localedge_numpy
    #    return indices
    

    #@profile
    def make_localedge(self, nodes_device, connect):
        nodes = nodes_device.to('cpu').numpy()
        edges_index = self.make_localedge_numpy(nodes, connect)
        edges_torch = torch.tensor(edges_index, dtype=torch.int64).to(DEVICE) # for Graph SAGE int64 (not int32)
        return edges_torch
    
    #@profile
    def make_localedge_numpy(self, nodes_numpy, connect):
        node_to_idx = self.make_node_idx(nodes_numpy)
        #node_to_idx = {node: idx for idx, node in enumerate(nodes_numpy)} 
        nodes_set = set(nodes_numpy)  # Convert to a set for faster lookup
        edges_from = []
        edges_to = []

        for start_node in nodes_numpy:
            start_idx = node_to_idx[start_node]
            for dest_node in connect[start_node]:
                if dest_node in nodes_set:        
                    dest_idx = node_to_idx[dest_node]
                    edges_from.append(start_idx)
                    edges_to.append(dest_idx)

        return [edges_from, edges_to]
    
    #@profile
    def make_gnn_dataset(self, nodes_info, nodes_float, connect):
        edges_index = self.make_localedge(nodes_float, connect)
        return Data(x=nodes_info, edge_index=edges_index)  
    
    def find_indices(self, abst_list, nodes_device):
        indices = []
        nodes_numpy = nodes_device.to('cpu').numpy()
        nodes_to_idx = self.make_node_idx(nodes_numpy)
        #nodes_to_idx = {node: idx for idx, node in enumerate(nodes_numpy)} # ここのせいで、最終的に抜き出しのindexが-1になっている注意
        #nodes_set = set(nodes_numpy)
        indices = [nodes_to_idx[abst] for abst in abst_list]
        return indices
    
    def make_node_idx(self, nodes_numpy):
        ret_dict = {}
        for idx, node in enumerate(nodes_numpy):
            if node not in ret_dict:
                ret_dict[node] = idx
        return ret_dict
        #return {node: idx for idx, node in enumerate(nodes_numpy)}# ここのせいで、最終的に抜き出しのindexが-1になっている注意
    
    #@profile
    def forward(self, x):
        #### Graph #####
        #print(x.shape)
        batch_num = x.shape[0]
        x_reshape = torch.reshape(x, (batch_num, KNN_AGENTS, -1))
        #print(x_reshape.shape)
        x_split = torch.split(x_reshape, (self.startnode_dims,self.destnode_dims,self.fromnodes,self.tonodes, self.elsedim), dim=2)
        x_start_infos = torch.reshape(x_split[0], (batch_num, KNN_AGENTS, self.fromnodes, -1))
        x_dest_infos = torch.reshape(x_split[1], (batch_num, KNN_AGENTS, self.tonodes, -1))
        x_start_graphs = x_split[2].clone().detach().to(torch.int32)
        x_dest_graphs = x_split[3].clone().detach().to(torch.int32)
        x_elses = x_split[4]

        x_feat = []
        x_start_nodes_zero = x_start_graphs[:,:,0].to('cpu').numpy()
        x_dest_nodes_zero = x_dest_graphs[:,:,0].to('cpu').numpy()

        x_all_start_infos = x_start_infos[:,:,:,NODE_DATA_EACH_DIM:NODE_DATA_DIM]
        x_all_dest_infos = x_dest_infos[:,:,:,NODE_DATA_EACH_DIM:NODE_DATA_DIM]
        x_each_start_infos = x_start_infos[:,:,:,0:NODE_DATA_EACH_DIM]
        x_each_dest_infos = x_dest_infos[:,:,:,0:NODE_DATA_EACH_DIM]

        #print(f"all_start:{x_all_start_infos[0,0,:,:]}")
        #print(f"all_dest:{x_all_dest_infos[0,0,:,:]}")
        #print(f"else:{x_elses[0,0,:]}")
        #print(f"each_start:{x_each_start_infos[0,0,:,:]}")
        #print(f"each_dest:{x_each_dest_infos[0,0,:,:]}")
        
        #print(x_all_start_infos.shape)
        #print(x_start_infos.shape)
        datasets = []
        for batch in range(batch_num):
            #x_start_info = x_start_infos[batch]
            #x_dest_info = x_dest_infos[batch]
            x_start_graph = x_start_graphs[batch]
            x_dest_graph = x_dest_graphs[batch]
            
            x_all_start_info = x_all_start_infos[batch]
            x_all_dest_info = x_all_dest_infos[batch]
            
            #print(x_all_start_info.shape)
            #print(x_all_dest_info.shape)
            x_graph = torch.cat([x_start_graph, x_dest_graph], dim=1)
            x_info = torch.cat([x_all_start_info, x_all_dest_info], dim=1)

            x_graph = torch.reshape(x_graph, (KNN_AGENTS*(NUM_NODES_OF_LOCAL_GRAPH_START+NUM_NODES_OF_LOCAL_GRAPH_DEST),))
            x_info = torch.reshape(x_info, (KNN_AGENTS*(NUM_NODES_OF_LOCAL_GRAPH_START+NUM_NODES_OF_LOCAL_GRAPH_DEST), -1))
            #print(x_graph.shape)
            #print(x_info.shape)
            
            datasets.append(self.make_gnn_dataset(x_info, x_graph, CONNECTDICT))

        dataset_loader = DataLoader(datasets, batch_size=len(datasets), shuffle=False)
        x_infos = []
        for data in dataset_loader:
            #print(data.x.shape)
            #print(data.edge_index)
            x_data = self.state_graph_l1(data.x, data.edge_index)
            x_data = F.relu(x_data)
            x_infos.append(x_data)
        #print("graph")
        x_all_infos_after_gnn = torch.cat(x_infos)
        x_all_infos_after_gnn = x_all_infos_after_gnn.reshape((batch_num, KNN_AGENTS*(self.fromnodes+self.tonodes), -1))
        #print(x_all_infos_after_gnn.shape)
        
        for batch in range(batch_num):
            x_all_info = x_all_infos_after_gnn[batch]
            x_start_graph = x_start_graphs[batch]
            x_dest_graph = x_dest_graphs[batch]
            x_else = x_elses[batch]
            x_each_start_info = x_each_start_infos[batch]
            x_each_dest_info = x_each_dest_infos[batch]

            x_start_node_zero = x_start_nodes_zero[batch]
            x_dest_node_zero = x_dest_nodes_zero[batch]
            
            x_graph = torch.cat([x_start_graph, x_dest_graph], dim=1)
            x_graph = torch.reshape(x_graph, (KNN_AGENTS*(NUM_NODES_OF_LOCAL_GRAPH_START+NUM_NODES_OF_LOCAL_GRAPH_DEST),)).to('cpu')
            
            x_tmp_feat = []
            for robot in range(KNN_AGENTS):
                x_start_graph_robot = x_start_graph[robot]
                x_dest_graph_robot = x_dest_graph[robot]
                x_else_robot = x_else[robot]
                x_start_node_zero_robot = x_start_node_zero[robot]
                x_dest_node_zero_robot = x_dest_node_zero[robot]

                x_each_start_info_robot = x_each_start_info[robot]
                x_each_dest_info_robot = x_each_dest_info[robot]
                
                x_each_info_robot = torch.cat([x_each_start_info_robot, x_each_dest_info_robot], dim=0)
                
                x_graph_robot = torch.cat([x_start_graph_robot, x_dest_graph_robot], dim=0)
                abst_list = ACTION_LIST[x_start_node_zero_robot][:]
                abst_list.append(x_dest_node_zero_robot)
                #print(abst_list)
                #print(x_graph)
                #print(x_each_start_info_robot.shape)
                #print(x_all_info.shape)
                rows = self.find_indices(abst_list, x_graph)
                
                rows_robot =self.find_indices(abst_list, x_graph_robot)

                x_tmp_feat.append(torch.cat([torch.flatten(x_all_info[rows]), x_else_robot, torch.flatten(x_each_info_robot[rows_robot])]))
                #if robot == 0 and batch==0:
                #    print(torch.cat([torch.flatten(x_all_info[rows]), x_else_robot, torch.flatten(x_each_info_robot[rows_robot])]))
                #print("done")
            x_tmp_feat = torch.cat(x_tmp_feat)
            x_feat.append(x_tmp_feat)


        x = torch.cat(x_feat)
        x = torch.reshape(x, (batch_num, -1))

        #### Graph End #####

        x = F.relu(self.l1(x))
        if DIM2:
            x = F.relu(self.l2(x))

        if DUELINGMODE:
            adv = self.ladv_last(x)
            v = self.lv_last(x)
            averagea = adv.mean(1, keepdim=True)
            return v.expand(-1, self.action_dims) + (adv - averagea.expand(-1, self.action_dims))
        else:
            x = self.l_last(x)
            return x
    
    """
    def forward(self, x):
        #### Graph #####
        #print(x.shape)
        batch_num = x.shape[0]
        x_reshape = torch.reshape(x, (batch_num, KNN_AGENTS, -1))
        #print(x_reshape.shape)
        x_split = torch.split(x_reshape, (self.startnode_dims,self.destnode_dims,self.fromnodes,self.tonodes, self.elsedim), dim=2)
        x_start_infos = torch.reshape(x_split[0], (batch_num, KNN_AGENTS, self.fromnodes, -1))
        x_dest_infos = torch.reshape(x_split[1], (batch_num, KNN_AGENTS, self.tonodes, -1))
        x_start_graphs = x_split[2].clone().detach().to(torch.int32)
        x_dest_graphs = x_split[3].clone().detach().to(torch.int32)
        x_elses = x_split[4]

        x_feat = []
        x_start_nodes_zero = x_start_graphs[:,:,0].to('cpu').numpy()
        x_dest_nodes_zero = x_dest_graphs[:,:,0].to('cpu').numpy()

        #print(x_start_infos.shape)
        datasets = []
        for batch in range(batch_num):
            x_start_info = x_start_infos[batch]
            x_dest_info = x_dest_infos[batch]
            x_start_graph = x_start_graphs[batch]
            x_dest_graph = x_dest_graphs[batch]
            for robot in range(KNN_AGENTS):
                x_start_info_robot = x_start_info[robot]
                x_dest_info_robot = x_dest_info[robot]
                x_start_graph_robot = x_start_graph[robot]
                x_dest_graph_robot = x_dest_graph[robot]

                #print(x_start_info_robot.shape)
                #print(x_start_graph_robot.shape)
                x_info = torch.cat([x_start_info_robot, x_dest_info_robot], dim=0)
                x_graph = torch.cat([x_start_graph_robot, x_dest_graph_robot], dim=0)
                #print(x_info.shape)
                #print(x_graph.shape)

                datasets.append(self.make_gnn_dataset(x_info, x_graph, CONNECTDICT))
        dataset_loader = DataLoader(datasets, batch_size=len(datasets), shuffle=False)
        x_infos = []
        for data in dataset_loader:
            #print(data.x.shape)
            #print(data.edge_index)
            x_data = self.state_graph_l1(data.x, data.edge_index)
            x_data = F.relu(x_data)
            x_infos.append(x_data)
        
        x_infos_after_gnn = torch.cat(x_infos)
        x_infos_after_gnn = x_infos_after_gnn.reshape((batch_num, KNN_AGENTS, self.fromnodes+self.tonodes, -1))

        for batch in range(batch_num):
            x_info = x_infos_after_gnn[batch]
            x_start_graph = x_start_graphs[batch]
            x_dest_graph = x_dest_graphs[batch]
            x_else = x_elses[batch]

            x_start_node_zero = x_start_nodes_zero[batch]
            x_dest_node_zero = x_dest_nodes_zero[batch]

            x_tmp_feat = []
            for robot in range(KNN_AGENTS):
                x_info_robot = x_info[robot]
                x_start_graph_robot = x_start_graph[robot]
                x_dest_graph_robot = x_dest_graph[robot]
                x_else_robot = x_else[robot]
                x_start_node_zero_robot = x_start_node_zero[robot]
                x_dest_node_zero_robot = x_dest_node_zero[robot]

                x_graph_robot = torch.cat([x_start_graph_robot, x_dest_graph_robot], dim=0)

                abst_list = ACTION_LIST[x_start_node_zero_robot][:]
                abst_list.append(x_dest_node_zero_robot)
                #print(abst_list)
                rows = self.find_indices(abst_list, x_graph_robot)
                x_tmp_feat.append(torch.cat([torch.flatten(x_info_robot[rows]), x_else_robot]))

            x_tmp_feat = torch.cat(x_tmp_feat)
            x_feat.append(x_tmp_feat)

        x = torch.cat(x_feat)
        x = torch.reshape(x, (batch_num, -1))

        #### Graph End #####

        x = F.relu(self.l1(x))
        if DIM2:
            x = F.relu(self.l2(x))

        if DUELINGMODE:
            adv = self.ladv_last(x)
            v = self.lv_last(x)
            averagea = adv.mean(1, keepdim=True)
            return v.expand(-1, self.action_dims) + (adv - averagea.expand(-1, self.action_dims))
        else:
            x = self.l_last(x)
            return x
        """
#"""